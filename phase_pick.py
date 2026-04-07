"""
1分単位の連続波形ファイルを読み込み、
15秒オーバーラップの4つの30秒窓に分割してモデル推論を行い、
重複区間を平均して再結合した後に P 相・S 相のピーク時刻を出力するスクリプト。

想定フロー:
1. 1分ファイルとその次の1分ファイルを読み込む
2. 各観測点について 60 秒波形を構成する
3. 30 秒窓を 15 秒ずつずらして 4 本作る
4. 各窓を z-score 正規化してモデルに入力する
5. 各窓の予測結果を時間方向に再結合する
6. P/S 確率時系列のピークを検出してテキスト出力する
"""

import os
import sys
import glob
import math
import datetime

import numpy as np
from scipy.signal import find_peaks
#import matplotlib.pyplot as plt

from win2ndarray import extract_waveform_metadata

import torch
import torch.nn as nn
from einops import rearrange

def convert_datetime(s):
    """
    ファイル名のような時刻文字列を np.datetime64 に変換する。

    想定入力:
        'YYMMDDHH.MM...' の先頭部分を含む文字列
    例:
        '26040112.34' -> np.datetime64('2026-04-01T12:34')

    注意:
        秒以下はここでは扱っていない。
    """
    # 文字列を分解
    year  = "20" + s[0:2]
    month = s[2:4]
    day   = s[4:6]
    hour  = s[6:8]
    minute = s[9:11]

    # ISO8601形式に整形
    iso_str = f"{year}-{month}-{day}T{hour}:{minute}"

    # np.datetime64 に変換
    np_dt = np.datetime64(iso_str, 'ms')
    return np_dt

def convert_str(dt):
    """
    np.datetime64 をファイル名形式 'YYMMDDHH.MM' に近い文字列へ戻す。

    例:
        np.datetime64('2011-03-08T22:24') -> '11030822.24'
    """
    s = dt.astype(str)          # "2011-03-08T22:24"

    # "11030822.24" に変換
    s = s.replace("-", "").replace("T", "").replace(":", "")
    s = s[2:8] + s[8:10] + "." + s[10:12]# + s[12:14]
    return s

def segmentation(data, lapse_time, start, end):
    """
    1 本の波形から [start, end) の時間範囲だけを切り出す。

    Parameters
    ----------
    data : ndarray, shape (3, T)
        3成分波形
    lapse_time : ndarray, shape (T,)
        各サンプルの絶対時刻
    start, end : np.datetime64
        切り出し区間
    """
    seg = data[:, (lapse_time>=start) & ((lapse_time<end))]
    return seg

def segmentation2(wave_f, wave_s, lapse_time_f, lapse_time_s, start, end):
    """
    2つの連続する 1 分波形をまたぐ区間を切り出すための関数。

    例:
        45秒〜75秒のように、前半は first minute, 後半は second minute
        から取る必要がある窓を結合する。
    """
    A = wave_f[:, (lapse_time_f>=start)]
    B = wave_s[:, (lapse_time_s<end)]
    return np.hstack([A, B])

def zscore(A):
    """
    各成分ごとに z-score 正規化する。

    入力 shape:
        (3, T)
    出力 shape:
        (3, T)

    標準偏差 0 の場合は 1 に置き換えて 0 除算を防ぐ。
    """
    mean = np.mean(A, axis=1, keepdims=True)
    std = np.std(A, axis=1, keepdims=True)
    # 0 除算を避ける
    std_safe = np.where(std == 0, 1, std)
    return (A - mean) / std_safe

def convert_wave(data, st, length, date):
    """
    観測点 st の 3 成分波形を長さ length にそろえて返す。

    元データが length より短い場合は、
    - 先頭時刻が期待時刻 date と一致しない → 前側を 0 埋め
    - 先頭時刻が date と一致する         → 後側を 0 埋め

    Parameters
    ----------
    data : dict
        extract_waveform_metadata が返す観測点ごとの辞書
    st : str
        観測点名
    length : int
        期待サンプル長（ここでは 100 Hz × 60 s = 6000）
    date : np.datetime64
        その 1 分波形の基準開始時刻
    """
    lapse_time = data[st]['time']
        
    wave = np.zeros((3, length))
    
    ud = data[st]['U']
    ns = data[st]['N']
    ew = data[st]['E']
    
    if len(lapse_time) == length:
        wave[0,:] = ud 
        wave[1,:] = ns 
        wave[2,:] = ew
    else:
        if lapse_time[0] != date:
            wave[0,length - len(lapse_time):] = ud 
            wave[1,length - len(lapse_time):] = ns 
            wave[2,length - len(lapse_time):] = ew 
        else:
            wave[0,:len(lapse_time)] = ud
            wave[1,:len(lapse_time)] = ns
            wave[2,:len(lapse_time)] = ew 
            
    return wave

def file_name(dt):
    """
    np.datetime64 を 'YYMMDD.HHMMSS' 形式の文字列へ変換する。

    これは detect_AT や再結合処理で各窓の開始時刻を扱うために使う。
    """
    s = np.datetime_as_string(dt, unit='s')  # '2011-03-15T14:58:00'
    fn = f"{s[2:4]}{s[5:7]}{s[8:10]}.{s[11:13]}{s[14:16]}{s[17:19]}"
    return fn

def detect_AT(pred, st, sf, date, mode, dire_path, th, start_year, start_month):
    """
    予測確率時系列 pred からピークを検出し、到達候補時刻をテキスト出力する。

    Parameters
    ----------
    pred : ndarray, shape (T,)
        ある観測点・ある相（P or S）の確率時系列
    st : str
        観測点名
    sf : float
        サンプリング間隔 [s]。100 Hz なら 0.01
    date : str
        開始時刻文字列 'YYMMDD.HHMMSS'
    mode : str
        'P' または 'S'
    dire_path : str
        出力先ディレクトリ
    th : float
        ピーク検出閾値
    start_year, start_month : str
        基準時刻（その月の 1 日 00:00:00）を作るために使用

    出力形式:
        CT.{station}.{P/S}.txt に
        [基準時刻からの秒, ピーク値, 0.0]
        を追記する。
    """
    
    reference_time = datetime.datetime(year=int(start_year),
                                       month=int(start_month),
                                       day=1)
    
    year  = 2000 + int(date[:2])   # 必要なら 1900年代の扱いに調整
    month = int(date[2:4])
    day   = int(date[4:6])
    hour  = int(date[7:9])         
    minute= int(date[9:11])
    second= int(date[11:13])
    
    # print(year, month, day, hour, minute, second)
    
    start_time = datetime.datetime(year=year,
                                   month=month,
                                   day=day,
                                   hour=hour,
                                   minute=minute,
                                   second=int(second))
    
    idxs, probs = find_peaks(pred, distance=int(1/sf)//2, height=th)
    
    for i in range(len(idxs)):
        if probs['peak_heights'][i] >= th:
            
            file_path = f'{dire_path}/CT.{st}.{mode}.txt'
            
            f = open(file_path, 'a')
            
            ad = datetime.timedelta(seconds=idxs[i]*sf)
            
            time = start_time + ad

            delta = time - reference_time
            
            if sf == 0.004:
                out = "{:.03f} {:} {:}\n".format(delta.total_seconds(), round(probs['peak_heights'][i],3), 0.0)
            else:
                out = "{:.02f} {:} {:}\n".format(delta.total_seconds(), round(probs['peak_heights'][i],3), 0.0)
            f.write(out)
            f.close()
        
    return

def merge_timeseries(data_list, start_times, fs):
    """
    重なりをもつ複数の時系列を、開始時刻情報を使って 1 本に再結合する。

    重複区間は単純平均する。

    Parameters
    ----------
    data_list : list of ndarray
        各窓の時系列データ
    start_times : list of datetime.datetime
        各窓の開始時刻
    fs : int
        サンプリング周波数 [Hz]

    Returns
    -------
    t0 : datetime.datetime
        再結合後時系列の開始時刻
    t : ndarray, shape (N,)
        t0 からの相対時刻 [s]
    y : ndarray, shape (N,)
        再結合後の時系列（欠測は NaN）
    """
    assert len(data_list) == len(start_times) and len(data_list) > 0
    data_list = [np.asarray(x) for x in data_list]
    L = [len(x) for x in data_list]
    dt = 1.0 / fs

    t0 = min(start_times)
    t_end = max(
        st + datetime.timedelta(seconds=(n - 1) * dt)
        for st, n in zip(start_times, L)
    )
    N_out = int(round((t_end - t0).total_seconds() * fs)) + 1  # 両端含む

    acc = np.zeros(N_out, dtype=float)
    w   = np.zeros(N_out, dtype=int)

    for x, st in zip(data_list, start_times):
        start_idx = int(round((st - t0).total_seconds() * fs))
        end_idx   = start_idx + len(x)
        acc[start_idx:end_idx] += x
        w[start_idx:end_idx]   += 1

    y = np.full(N_out, np.nan, dtype=float)
    m = (w > 0)
    y[m] = acc[m] / w[m]

    t = np.arange(N_out) / fs
    return t0, t, y

def parse_yymmdd_hhmmss(s: str) -> datetime.datetime:
    """
    'YYMMDD.HHMMSS' を datetime に変換する。

    例:
        '130107.222915' -> 2013-01-07 22:29:15
    """
    if len(s) != 13 or s[6] != '.':
        raise ValueError(f"Invalid format: {s} (expected 'YYMMDD.HHMMSS')")
    yy   = int(s[0:2])
    mm   = int(s[2:4])
    dd   = int(s[4:6])
    HH   = int(s[7:9])
    MM   = int(s[9:11])
    SS   = int(s[11:13])
    year = 2000 + yy
    return datetime.datetime(year, mm, dd, HH, MM, SS)

def merge_timeseries_from_strings(data_list, start_strs, fs):
    """
    開始時刻が文字列で与えられている場合のラッパー関数。

    Parameters
    ----------
    start_strs : list of str
        ['YYMMDD.HHMMSS', ...]
    """
    start_times = [parse_yymmdd_hhmmss(s) for s in start_strs]
    return merge_timeseries(data_list, start_times, fs)

def main(fn, model_FT, device):
    """
    1つの 1 分ファイルに対するメイン処理。

    処理内容:
    1. 現在ファイルと次の 1 分ファイルを読む
    2. 各観測点の 60 秒波形を作る
    3. 15 秒オーバーラップの 30 秒窓を 4 本作る
    4. 各窓をモデル推論する
    5. 4 本の予測確率を再結合する
    6. P/S のピークを検出してファイル出力する
    """
    
    if os.path.getsize(fn) == 0:
        return
    
    # print(fn)
    
    first_fn = fn
    first_date = convert_datetime(os.path.basename(fn))
    
    second_date = first_date + np.timedelta64(1, "m")
    sfn = os.path.basename(fn).replace(first_fn.split('/')[-1], convert_str(second_date))
    
    path_dir = os.path.dirname(fn)
    second_fn = f'{path_dir}/{sfn}'
    
    chf_f = 'station_HAGI.list'
    waveform_info_df_f, waveform_dict_f = extract_waveform_metadata(first_fn, chf_f)
    
    chf_s = 'station_HAGI.list'
    waveform_info_df_s, waveform_dict_s = extract_waveform_metadata(second_fn, chf_s)
    
    seg1_start = first_date + np.timedelta64(0, "ms")
    seg1_end = first_date + np.timedelta64(30000, "ms")
    
    seg2_start = first_date + np.timedelta64(15000, "ms")
    seg2_end = first_date + np.timedelta64(45000, "ms")
    
    seg3_start = first_date + np.timedelta64(30000, "ms")
    seg3_end = first_date + np.timedelta64(60000, "ms")
    
    seg4_start = first_date + np.timedelta64(45000, "ms")
    seg4_end = second_date + np.timedelta64(15000, "ms")
    
    seg1s = []
    seg2s = []
    seg3s = []
    seg4s = []
    
    stations = []
    
    length = 100 * 60
    
    master_lapse_time_f = first_date + np.arange(length) * np.timedelta64(10, 'ms')
    master_lapse_time_s = second_date + np.arange(length) * np.timedelta64(10, 'ms')
    
    for st in waveform_dict_f.keys():
        
        try:
        
            wave_f = convert_wave(waveform_dict_f, st, length, first_date)
            wave_s = convert_wave(waveform_dict_s, st, length, second_date)
            
        
            seg1 = segmentation(wave_f, master_lapse_time_f, seg1_start, seg1_end)
            seg2 = segmentation(wave_f, master_lapse_time_f, seg2_start, seg2_end)
            seg3 = segmentation(wave_f, master_lapse_time_f, seg3_start, seg3_end)
            seg4 = segmentation2(wave_f, wave_s, master_lapse_time_f, master_lapse_time_s, seg4_start, seg4_end)
            
            seg1s.append(zscore(seg1))
            seg2s.append(zscore(seg2))
            seg3s.append(zscore(seg3))
            seg4s.append(zscore(seg4))
            
            # print(seg1.shape)
            
            stations.append(st)
        except:
            continue
        
        
    input1 = torch.from_numpy(np.stack(seg1s).astype('float32'))
    input2 = torch.from_numpy(np.stack(seg2s).astype('float32'))
    input3 = torch.from_numpy(np.stack(seg3s).astype('float32'))
    input4 = torch.from_numpy(np.stack(seg4s).astype('float32'))
    
    model_FT.to(device)
    model_FT.eval()
    
    with torch.no_grad():
        pred1_FT = model_FT(input1.to(device)).detach().cpu().numpy()
        pred2_FT = model_FT(input2.to(device)).detach().cpu().numpy()
        pred3_FT = model_FT(input3.to(device)).detach().cpu().numpy()
        pred4_FT = model_FT(input4.to(device)).detach().cpu().numpy()
   
    fn_seg1 = file_name(seg1_start)
    fn_seg2 = file_name(seg2_start)
    fn_seg3 = file_name(seg3_start)
    fn_seg4 = file_name(seg4_start)
    
    year = f'20{os.path.basename(fn)[:2]}'
    yymm = f'{os.path.basename(fn)[:4]}'
    month = f'{os.path.basename(fn)[2:4]}'
    
    dire_path = f'./pred_res'
    os.makedirs(dire_path, exist_ok=True)
    
    pth = 0.5
    sth = 0.5
    
    start_strs = [fn_seg1,
                  fn_seg2,
                  fn_seg3,
                  fn_seg4,]

    
    for k in range(len(stations)):
        
        t0, t, pred_P = merge_timeseries_from_strings([pred1_FT[k][0,:], pred2_FT[k][0,:], pred3_FT[k][0,:], pred4_FT[k][0,:]],
                                                 start_strs,
                                                 fs=100)
        
        t0, t, pred_S = merge_timeseries_from_strings([pred1_FT[k][1,:], pred2_FT[k][1,:], pred3_FT[k][1,:], pred4_FT[k][1,:]],
                                                 start_strs,
                                                 fs=100)
        
        # print(month)
        
        detect_AT(pred_P, stations[k], 0.01, fn_seg1, 'P', dire_path, pth, year, month)
        detect_AT(pred_S, stations[k], 0.01, fn_seg1, 'S', dire_path, sth, year, month)
    
    return 

from SegPhase.model_str import Model

device = 'cuda:2'
model = Model(in_length=100*30, in_channels=3, class_num=3,strides=[3,2,2], kernel_size=3).to(device)
state_dict = torch.load('SegPhase/best_model.pth', map_location=device, weights_only=True)
model.load_state_dict(state_dict)


files = sorted(glob.glob(f'./trg_20250401_HAGI/*'))
total = len(files)
print(total)

for c, fn in enumerate(files[:-1], start=1):
    
    print(f'{c}/{total} - {os.path.basename(fn)}', flush=True)
 
    # try:
    main(fn, model, device)
    # break

    # except KeyboardInterrupt:
    #     # Ctrl+C で即停止
    #     print("\n[INTERRUPT] Ctrl+C を受け取りました。処理を中断します。")
    #     break

    # except Exception as e:
    #     # 失敗したファイルだけ記録
    #     with open(f'miss.list', 'a') as f:
    #             f.write(f"{fn}\n")
            
