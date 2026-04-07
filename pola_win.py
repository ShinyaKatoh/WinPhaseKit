"""
このスクリプトは、イベント情報と各観測点の P/S ピックを入力として、
対応する波形データを読み込み、P 波初動極性の推定と位相情報の整理を行い、
最終的に HYPOMH 用の入力ファイルを作成するための処理をまとめたものです。

主な処理の流れ
1. イベント一覧とピック情報を読み込む
2. イベント時刻に対応する WIN 波形を観測点ごとに取得する
3. 各観測点について P/S 到達時刻の index を求める
4. P 到達付近 256 サンプルを PoViT モデルへ入力して初動極性を推定する
5. 位相情報・極性情報・振幅情報をまとめて HYPOMH 用フォーマットへ出力する

"""
from __future__ import annotations

import os
import sys
import glob
import subprocess
import shutil
import numpy as np
import torch
import pandas as pd
from win2ndarray import extract_each_station_waveform_metadata
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import subprocess
import datetime
from scipy.signal import butter, filtfilt

from PoViT.model_str import Model

from pathlib import Path

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model100_P = Model(in_length=256, kernel_size=16, ds_kernel_size=9, ff_kernel_size=9, seg_kernel_size=9, stride=1, head_num=4, emb_dim=64, num_blocks=7, dropout_ratio=0.3)
model100_P.load_state_dict(torch.load('./PoViT/model_100Hz.pth', map_location=device, weights_only=True))
model100_P.eval()
model100_P.to(device)

def _to_float(x: str) -> float:
    """
    文字列を float に変換する。

    inf, -inf, nan, -nan のような特殊値も受け付け、
    変換できない場合は NaN を返す。

    Parameters
    ----------
    x : str
        数値文字列。

    Returns
    -------
    value : float
        変換後の浮動小数点値。変換できない場合は NaN。
    """
    try:
        return float(x)
    except ValueError:
        return float("nan")
    


def read_event_pick_file(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    イベントヘッダ行と CT 行を含むテキストファイルを読み取る。

    ファイル中のイベント情報を events_df、各 station のピック情報を
    picks_df に分けて整理して返す。

    Parameters
    ----------
    path : str
        入力ファイルのパス。

    Returns
    -------
    events_df : pandas.DataFrame
        1 行 1 イベントの表。
    picks_df : pandas.DataFrame
        1 行 1 ピックの表。event_id により events_df と対応付けられる。
    """
    events = []
    picks = []

    current_event_id = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            cols = line.split()

            # ---- イベントヘッダ行（先頭が整数）----
            if cols[0].isdigit():
                # 例:
                # 1 2011 03 09 03:16:21.910 702981.910 0.0949 -43.5991 172.5840 0.00 -inf -nan 4 4 8 4 328.38
                current_event_id = int(cols[0])

                # 最低限の主要情報は列位置で確保しつつ、rawも残す
                y, mo, d = int(cols[1]), int(cols[2]), int(cols[3])
                time_str = cols[4]

                ev = {
                    "event_id": current_event_id,
                    "date": f"{y:04d}-{mo:02d}-{d:02d}",
                    "time": time_str,
                    "origin_sec": _to_float(cols[5]),
                    "val_6": _to_float(cols[6]),
                    "lat": _to_float(cols[7]),
                    "lon": _to_float(cols[8]),
                    "dep": _to_float(cols[9]),
                    "val_10": _to_float(cols[10]),
                    "val_11": _to_float(cols[11]),
                    "i12": int(cols[12]),
                    "i13": int(cols[13]),
                    "i14": int(cols[14]),
                    "i15": int(cols[15]),
                    "val_16": _to_float(cols[16]),
                    "raw_line": line,
                }
                events.append(ev)
                continue

            # ---- ピック行（CTで始まる）----
            if cols[0] == "CT":
                # CT行がイベント開始前に出るならフォーマット異常なので無視（必要なら raise に変更）
                if current_event_id is None:
                    continue

                # 例:
                # CT DP.NZ5 P 702989.9600 8.0499 0.00e+00 0.0285 0.3690 83.0602
                pk = {
                    "event_id": current_event_id,
                    "net": cols[0],            # "CT"
                    "station": cols[1],        # "DP.NZ5"
                    "phase": cols[2],          # "P" or "S"
                    "t_abs": _to_float(cols[3]),
                    "t_rel": _to_float(cols[4]),
                    "col6": _to_float(cols[5]),
                    "col7": _to_float(cols[6]),
                    "prob": _to_float(cols[7]),
                    "col9": _to_float(cols[8]),
                    "raw_line": line,
                }
                picks.append(pk)
                continue

            # その他の行は無視（必要ならログ化）
            continue

    events_df = pd.DataFrame(events)
    picks_df = pd.DataFrame(picks)

    # 任意：ソートして扱いやすく
    if not events_df.empty:
        events_df = events_df.sort_values(["event_id"]).reset_index(drop=True)

    if not picks_df.empty:
        picks_df = picks_df.sort_values(["event_id", "station", "phase"]).reset_index(drop=True)

    return events_df, picks_df

def round_datetime64_to_ms(t):
    """
    np.datetime64 をミリ秒単位に四捨五入する。

    NaT はそのまま datetime64[ms] に変換して返す。

    Parameters
    ----------
    t : np.datetime64
        変換対象の時刻。

    Returns
    -------
    t_ms : np.datetime64
        ミリ秒単位に丸めた時刻。
    """
    if np.isnat(t):
        return t.astype("datetime64[ms]")

    ns = t.astype("datetime64[ns]").astype(np.int64)
    unit = 1_000_000  # 1 ms = 1,000,000 ns
    half = unit // 2

    if ns >= 0:
        ns_rounded = ((ns + half) // unit) * unit
    else:
        ns_rounded = ((ns - half) // unit) * unit

    # ここがポイント：numpy.int64 -> Python int にする
    return np.datetime64(int(ns_rounded), "ns").astype("datetime64[ms]")

def create_winf_path(t, root: str = "trg_20250401_HAGI"):
    """
    np.datetime64 を、指定ルールの WIN ファイルパスに変換する。

    出力形式は以下を想定する。
      root/YYMMDDHH.MM

    例:
      2011-03-09T03:48:10.247 -> .../11030903.48

    Parameters
    ----------
    t : np.datetime64
        変換元の時刻。
    root : str, optional
        WIN ファイルが保存されているルートディレクトリ。

    Returns
    -------
    win_filepath : str
        指定時刻に対応する WIN ファイルパス。
    """
    # 分までに丸め/切り捨てして "YYYY-MM-DDTHH:MM" を得る
    # (秒以下は不要なので minute へ落とす)
    s = str(t.astype("datetime64[m]"))  # e.g. "2011-03-09T03:48"

    date_part, hm_part = s.split("T")   # "2011-03-09", "03:48"
    y, mo, d = date_part.split("-")     # "2011","03","09"
    hh, mm = hm_part.split(":")         # "03","48"

    yymm = y[2:] + mo                   # "1103"
    yymmdd = y[2:] + mo + d             # "110309"
    fname = f"{yymmdd}{hh}.{mm}"        # "11030903.48"

    return f"{root}/{fname}"
    # return f"{root}/{fname}"

def zscore(waveform):
    """
    波形に Z スコア正規化を適用する。

    標準偏差が 0 の場合は、平均値のみを引いた波形を返す。

    Parameters
    ----------
    waveform : ndarray
        入力波形。

    Returns
    -------
    waveform_norm : ndarray
        正規化後の波形。
    """
    mean = np.mean(waveform)
    std = np.std(waveform)

    if std == 0:
        return waveform - mean  # 定数波形の場合

    return (waveform - mean) / std

def post_process(polas, ats):
    """
    極性分類結果と到達時刻確率から最終的な極性と代表到達時刻を求める。

    複数回の推論結果に対して、到達時刻はピーク位置の中央値、
    極性はクラス確率の中央値に基づいて決定する。

    Parameters
    ----------
    polas : ndarray
        極性クラス確率。shape はおおむね (n_trial, n_class)。
    ats : ndarray
        到達時刻確率系列。shape はおおむね (n_trial, 1, T)。

    Returns
    -------
    pola : str
        最終極性。'U', 'D', 'N' のいずれか。
    prob : float
        採用した極性クラスの代表確率。
    uq : float
        採用クラス確率の四分位範囲。
    at : float
        到達時刻ピーク index の中央値。
    """
    idxs = []
    probs = []
    
    for i in range(polas.shape[0]):
        idx, prob = find_peaks(ats[i][0,:], distance=100, height=0.1)
        
        if len(idx) > 0:
            
            idxs.append(idx[0])
            probs.append(prob['peak_heights'][0])
            
        else:
            idxs.append(0)
            probs.append(0)
    
    cla_median = np.median(polas, axis=0)
    
    pola_idx = np.argmax(cla_median)
    
    if pola_idx == 0:
        pola = 'U'
    elif pola_idx == 1:
        pola = 'D'
    elif pola_idx == 2:
        pola = 'N'
    
    Q1 = np.percentile(polas[:,pola_idx], 25)
    Q3 = np.percentile(polas[:,pola_idx], 75)
    out_IQR = Q3 - Q1
    
    return pola, cla_median[pola_idx], out_IQR, np.median(idxs)

def for_hypomh(origin_time, phase_dict, pola_dict, waveform_dict, waveform_info_df, ilat, ilon, idep, dire):
    """
    HYPOMH 用の入力ファイルを作成して外部プログラムを実行する。

    震源情報、station ごとの P/S 到達時刻、初動極性、振幅情報を
    所定フォーマットで書き出し、hypomh_hs5_jma を実行する。

    Parameters
    ----------
    origin_time : np.datetime64
        イベントの発震時刻。
    phase_dict : dict
        station ごとの P/S 到達時刻や確率、振幅を持つ辞書。
    pola_dict : dict
        station ごとの極性推定結果を持つ辞書。
    waveform_dict : dict
        波形辞書。
    waveform_info_df : pandas.DataFrame
        station 位置や標高を含むメタデータ表。
    ilat : float
        初期緯度。
    ilon : float
        初期経度。
    idep : float
        初期深さ [km]。
    dire : str
        出力先ディレクトリ。

    Returns
    -------
    None
        入力ファイル作成と外部実行のみ行う。
    """

    s = np.datetime_as_string(origin_time, unit='s')  # '2011-03-10T20:08:41'
    pn = f"{s[2:4]}{s[5:7]}{s[8:10]}.{s[11:13]}{s[14:16]}{s[17:19]}"   
    
    s2 = np.datetime_as_string(origin_time, unit="m")  # '2011-03-10T20:08'

    ref_dating = f"{s2[2:4]}/{s2[5:7]}/{s2[8:10]}"  # '11/03/10'
    ref_timing = s2[11:16]     

    start_time = origin_time.astype('datetime64[m]')
    
    out_inputdir = f'{dire}/res_pre_pick'
    out_outputdir = f'{dire}/res_pick/'
    
    # --- out_inputdir ---
    # if os.path.islink(out_inputdir) or os.path.isfile(out_inputdir):
    #     os.remove(out_inputdir)
    # elif os.path.isdir(out_inputdir):
    #     shutil.rmtree(out_inputdir)
    os.makedirs(out_inputdir, exist_ok=True)
    
    out_inputdir_f = f'{out_inputdir}/{pn}'

    # --- out_outputdir ---
    # if os.path.islink(out_outputdir) or os.path.isfile(out_outputdir):
    #     os.remove(out_outputdir)
    # elif os.path.isdir(out_outputdir):
    #     shutil.rmtree(out_outputdir)
    os.makedirs(out_outputdir, exist_ok=True)

    outf = open(out_inputdir_f, "w")

    head_p = "#p"
    head_s = "#s"

    name = "DL_Katoh"
    
    dt_now = datetime.datetime.now()
    today_date = str(dt_now.year)[2:] +"/"+ str(dt_now.month) + "/" + str(dt_now.day)
    today_time = str(dt_now.hour) + ":" + str(dt_now.minute) + ":" + str(dt_now.second)
    
    date_str = pn.split('.')[0]
    time_str = pn.split('.')[1]
    
    out1 = "{0:2s} {1:13s} {2:8s}\n".format(head_p, pn, name)

# FORMAT    3X,     I2,  I2,   I2,   1X,I2, I2
    # out2 = "{:}".format(second_line)
    out2 = f"#p {date_str[:2]} {date_str[2:4]} {date_str[4:]} {time_str[:2]} {time_str[2:4]} {time_str[4:]}\n"

    out3 = "{0:2s} {1:8s} {2:5s}                   {3:8s} {4:8s}\n".format(head_s, ref_dating, ref_timing, today_date, today_time)

    outf.write(out1)
    outf.write(out2)
    outf.write(out3)

    stations = sorted(list(phase_dict.keys()))

    for c, st in enumerate(stations):
        
        sf = 0.01
        
        if phase_dict[st]['P'] != 0:
            p_time = (phase_dict[st]['P'] - start_time) / np.timedelta64(1, 's')
            if sf == 0.01:
                p_accu = 0.02
            elif sf == 0.004:
                p_accu = 0.01
        else:
            p_time = 0.000
            p_accu = 0.000
            
        if phase_dict[st]['S'] != 0:
            s_time = (phase_dict[st]['S'] - start_time) / np.timedelta64(1, 's')
            if sf == 0.01:
                s_accu = 0.05
            elif sf == 0.004:
                s_accu = 0.03
        else:
            s_time = 0.000
            s_accu = 0.000
            
        lat = waveform_info_df[(waveform_info_df['Station'] == st) & (waveform_info_df['Component'] == 'U')]['Lat'].values[0]
        lon = waveform_info_df[(waveform_info_df['Station'] == st) & (waveform_info_df['Component'] == 'U')]['Lon'].values[0]
        dep = waveform_info_df[(waveform_info_df['Station'] == st) & (waveform_info_df['Component'] == 'U')]['Elv'].values[0]
        
        p_prob = phase_dict[st]['P_prob']
        s_prob = phase_dict[st]['S_prob']
        
        pola_prob = pola_dict[st]['prob']
        pola_UQ = pola_dict[st]['uq']
        diff = int(pola_dict[st]['at'])
        
        amp = phase_dict[st]['amp']
        
# HYPOMH FORMAT(3X,    A10,    1X,A1, F8.3,     F6.3,   F8.3,    F6.3,    F6.1,    E9.2,   F11.5,    F11.5,     I7,    F7.3,     F7.3)
        out4 = (
            "{0:<3s}"
            "{1:<10s}"
            "{2:>2s}"
            "{3:>8.3f}{4:>6.3f}{5:>8.3f}{6:>6.3f}"
            "{7:>6.1f}{8:9.2E}"
            "{9:>11.5f}{10:>11.5f}{11:>7}"
            "{12:>7.3f}{13:7.3f}"
            "{14:>7.3f}{15:>7.3f}"
            "{16:>7.3f}{17:>7.3f}{18:>5d}\n"
            ).format(
                '#s', 
                st, 
                pola_dict[st]['pola'], 
                p_time, p_accu, s_time, s_accu, 
                0.000, amp, 
                lat, lon, int(dep), 
                0.0, 0.0, 
                p_prob, s_prob, 
                pola_prob, pola_UQ, diff)
        outf.write(out4)
        
        # print(out4)

    out5 = "{0:2s}".format(head_s)

    outf.write(out5)

    outf.close()

    # subprocess.run(['./hypomh/hypomh_hs5_jma', out_inputdir_f, out_outputdir])

    return

def concat_waveform_dicts(*waveform_dicts, components=None, rebuild_time_step=True):
    """
    複数の waveform_dict を station/component ごとに連結して 1つにまとめる。

    Parameters
    ----------
    *waveform_dicts : dict
        waveform_dict を複数（例: A, B, C）渡す。
        期待する構造: d[station][component] = waveform(np.ndarray)
                     d[station]['time'] = times(np.ndarray of datetime64)
                     d[station]['time_step'] = time_step(np.ndarray)  (任意)
    components : list[str] or None
        連結対象の component を指定する場合（例 ['U','N','E']）。
        None の場合は dict に存在する component を自動収集（time/time_step は除外）。
    rebuild_time_step : bool
        True の場合、連結後の time から time_step を再計算して格納する。

    Returns
    -------
    merged : dict
        連結後の waveform_dict
    """
    if len(waveform_dicts) == 0:
        return {}

    # 全 station を収集
    stations = set()
    for d in waveform_dicts:
        if d is None:
            continue
        stations = set(d.keys())

    merged = {}

    for st in stations:
        merged[st] = {}

        # stationごとの component を決める
        if components is None:
            comps = set()
            for d in waveform_dicts:
                comps |= set(d.get(st, {}).keys())
            comps -= {"time", "time_step"}
        else:
            comps = set(components)

        # --- component 波形の連結 ---
        for comp in comps:
            arrs = []
            for d in waveform_dicts:
                a = d.get(st, {}).get(comp, None)
                if a is None:
                    continue
                a = np.asarray(a)
                if a.size == 0:
                    continue
                arrs.append(a)

            if len(arrs) == 0:
                continue

            merged[st][comp] = np.concatenate(arrs, axis=0)

        # --- time の連結 ---
        t_arrs = []
        for d in waveform_dicts:
            t = d.get(st, {}).get("time", None)
            if t is None:
                continue
            t = np.asarray(t)
            if t.size == 0:
                continue
            t_arrs.append(t)

        if len(t_arrs) > 0:
            merged[st]["time"] = np.concatenate(t_arrs, axis=0)

            if rebuild_time_step:
                t0 = merged[st]["time"][0]
                merged[st]["time_step"] = (merged[st]["time"] - t0) / np.timedelta64(1, "s")
        else:
            # time が無い場合、time_step も作らない
            pass

    return merged

def main(events_df, picks_df, dire):
    
    for event_id, grp in picks_df.groupby("event_id", sort=True):
        # if event_id < 24577:
        #     continue
        try:
            
            print('--------------------------------------------------------------------------------------------------------------------')
            print(event_id)
            
            pola_dict = {}
            phase_dict = {}
            
            row = events_df.loc[events_df["event_id"] == event_id].iloc[0]
            
            ilat = row['lat']
            ilon = row['lon']
            idep = row['dep']
            
            date_ = row["date"]
            time_ = row["time"]
            origin_time = np.datetime64(f"{date_}T{time_}", "ns")
            
            ## arrival_time
            t_info = picks_df[picks_df["event_id"] == event_id]
            
            all_stations = t_info["station"].to_numpy()
            uniq_stations = np.unique(all_stations)
            
            ## Waveform load
            winf_1 = create_winf_path(origin_time)
            chf_1 = 'station_HAGI.list'
            waveform_info_df_1, waveform_dict_1 = extract_each_station_waveform_metadata(winf_1, chf_1, uniq_stations)
            
            winf_2 = create_winf_path(origin_time + np.timedelta64(1, "m"))
            chf_2 = 'station_HAGI.list'
            waveform_info_df_2, waveform_dict_2 = extract_each_station_waveform_metadata(winf_2, chf_2, uniq_stations)
            
            waveform_dict = concat_waveform_dicts(waveform_dict_1, waveform_dict_2)
            
            for uniq_station in uniq_stations:
                    
                if uniq_station not in pola_dict:
                    pola_dict[uniq_station] = {'pola': 'N', 'prob': 0, 'uq':0, 'at':0}
                        
                if uniq_station not in phase_dict:
                    phase_dict[uniq_station] = {'P': 0.0, 'S': 0.0, 'P_prob':0.0, 'S_prob':0.0, 'P_idx':0, 'S_idx':0, 'amp':0}
                
                tmp_P = picks_df.loc[(picks_df["event_id"] == event_id) & (picks_df["phase"] == "P") & (picks_df["station"] == uniq_station), "t_rel"]
                plus_P = tmp_P.iloc[0] if not tmp_P.empty else 0.0
                
                tmp_S = picks_df.loc[(picks_df["event_id"] == event_id) & (picks_df["phase"] == "S") & (picks_df["station"] == uniq_station), "t_rel"]
                plus_S = tmp_S.iloc[0] if not tmp_S.empty else 0.0
                
                
                if plus_P != 0.0:
                    tp = round_datetime64_to_ms(origin_time + np.timedelta64(int(round(plus_P * 1e6)), "us"))
                    phase_dict[uniq_station]['P'] = tp
                    phase_dict[uniq_station]['P_prob'] = picks_df.loc[(picks_df["event_id"] == event_id) & (picks_df["phase"] == "P") & (picks_df["station"] == uniq_station), 'prob'].iloc[0]
                    phase_dict[uniq_station]['P_idx'] = np.abs(waveform_dict[uniq_station]['time'] - tp).argmin()
                else:
                    phase_dict[uniq_station]['P'] = 0.0
                    phase_dict[uniq_station]['P_prob'] = 0.0
                    phase_dict[uniq_station]['P_idx'] = 0
                    
                if plus_S != 0.0:
                    ts = round_datetime64_to_ms(origin_time + np.timedelta64(int(round(plus_S * 1e6)), "us"))
                    phase_dict[uniq_station]['S'] = ts
                    phase_dict[uniq_station]['S_prob'] = picks_df.loc[(picks_df["event_id"] == event_id) & (picks_df["phase"] == "S") & (picks_df["station"] == uniq_station), 'prob'].iloc[0]
                    phase_dict[uniq_station]['S_idx'] = np.abs(waveform_dict[uniq_station]['time'] -ts).argmin()
                else:
                    phase_dict[uniq_station]['S'] = 0.0
                    phase_dict[uniq_station]['S_prob'] = 0.0
                    phase_dict[uniq_station]['S_idx'] = 0
                    
                # Polarity Detection
                if phase_dict[uniq_station]['P_idx'] != 0:
                    if phase_dict[uniq_station]['S_idx'] != 0:
                        tud = waveform_dict[uniq_station]['time']
                        ud =  waveform_dict[uniq_station]['U']
                        amp = ud[phase_dict[uniq_station]['P_idx']:phase_dict[uniq_station]['S_idx']+500]
                        phase_dict[uniq_station]['amp'] = np.max(np.abs(amp - np.mean(amp)))
                    else:
                        tud = waveform_dict[uniq_station]['time']
                        ud =  waveform_dict[uniq_station]['U']
                        amp = ud[phase_dict[uniq_station]['P_idx']:phase_dict[uniq_station]['P_idx']+1000]
                        phase_dict[uniq_station]['amp'] = np.max(np.abs(amp - np.mean(amp)))
                        
                    wave = zscore(ud[phase_dict[uniq_station]['P_idx']-128:phase_dict[uniq_station]['P_idx']+128]).reshape(1,256).astype(np.float32)
                    input_wave = torch.from_numpy(np.repeat(wave[np.newaxis, :, :], 100, axis=0).astype('float32'))
                    
                    with torch.no_grad():
                        polas, ats = model100_P(input_wave.to(device))
                        
                    pola, prob, uq, at = post_process(polas.cpu(), ats.cpu())
                    
                    pola_dict[uniq_station]['pola'] = pola
                    pola_dict[uniq_station]['prob'] = prob
                    pola_dict[uniq_station]['uq'] = uq
                    pola_dict[uniq_station]['at'] = 128 - at
                    
                else:
                    phase_dict[uniq_station]['amp'] = 0.0
                    
                    pola_dict[uniq_station]['pola'] = 'N'
                    pola_dict[uniq_station]['prob'] = 0.0
                    pola_dict[uniq_station]['uq'] = 0.0
                    pola_dict[uniq_station]['at'] = 0.0
            
                
            for_hypomh(origin_time, phase_dict, pola_dict, waveform_dict, waveform_info_df_1, ilat, ilon, idep, dire)
        except KeyboardInterrupt:
            print("\nCtrl+C が押されたので終了します")
            break
        except Exception as e:
            print(f"エラー: {e}")
            continue
    return

if __name__ == '__main__':
    
    path = './pred_res/phase_sel.txt'
    
    events_df, picks_df = read_event_pick_file(path)
    print(events_df.head())
    print(picks_df.head())
    main(events_df, picks_df, os.path.dirname(path))
            
            
