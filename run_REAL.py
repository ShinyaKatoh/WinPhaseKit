
import numpy as np
import pandas as pd
import datetime


import glob
import subprocess

from multiprocessing import Pool
import time

def main(files):

    ## -D
    year = '2025'
    mon = '4'
    day = '1'
    
    ##-S
    # -S:(np0/ns0/nps0/npsboth0/std0/dtps/nrt/drt/nxd/rsel/ires])(int/int/int/int/double/double/double/[double/int])
    #       3/  2/   5/       2/ 0.5/ 0.2/1.5/0.5, 
    #       3/  2/   5/       2/ 0.5/ 0.2/1.5/0.5/0.5, 
    #       3/  2/   5/       2/ 0.5/ 0.2/1.5/0.5/0.5/   4/
    #       3/  2/   5/       1/ 0.5/ 0.2/1.5/0.5/0.5/   4/1
    # np0: threshold for number of P picks
    # ns0: threshold for number of S picks
    # nps0: threshold for total number of picks (P&S)
    # npsboth0: number of stations that recorded both P and S picks
    #           [It can significantly improve your association reliability, especially when you have a poor station coverage. 
    #           Along with np0, ns0, nps0, more picks/stations, more strict thresholds]
    # std0: standard deviation threshold (residual) 
    #       [Here residual is defined as the deviation of the origin times that were estimated from different picks (for the same event). 
    #       It is not the traditional RMS residual]
    # dtps: time threshold for S and P separation 
    #       [dtps is used to remove some false S picks. P picks may appear in S pick pool in real data when applying STA/LTA pickers. 
    #        S picks will not be used if ts-tp is less than dtps. Set dtps = 0 to turn this constraint off]
    # nrt: nrt*default time window (usually between 1 and 2) 
    #       [i.e., nrt*sqrt(tdx**2+ tdx**2+tdh**2)/vp0 for P time window or nrt*sqrt(tdx**2+tdx**2+tdh**2)/vs0 for S time window. 
    #       It accommodates the inaccuracy of velocity model (as well as pick uncertainty). Use larger nrt if velocity model is insufficient. 
    #       Note: nrt and tdx/tdh trade off! If you use a large grid size, please use small nrt, vice versa. 
    #       If your P and S t_dist curves are too narrow (clearly clipped), maybe your velocity model is too bad or the nrt is too small.]
    # drt: drt*default time window (usually < 1). 
    #       Remove associated picks < drt*P_window from the initiating pick pool. 
    #       Use it as small as possible if time affordable. Default is 0.5. Like the nrt, drt and tdx/tdh trade off. 
    #       Please use a small drt if you use a large grid size.
    # nxd: suspicious events with the nearest station > nxd*GCarc0 will be discarded.
    # rsel: tolerance multiplier; keep picks with residuals less than rsel*STD, and remove suspicious picks in large distance 
    #       (i.e., tpick should be smaller than tpick_median + 0.75*rsel*tpick_STD, 0.75 is fixed in code). 
    #       [rsel is used to remove picks with large residuals and large distance. 
    #       Default rsel is 4.0 (a large value, i.e., approximately turn this constraint off). 
    #       For example, if your final standard deviation is 0.1 sec, REAL will automatically remove those suspicious picks with residuals > 0.4 sec; 
    #       If you have large topography (i.e., station corr.), inaccurate velocity or picking uncertainty, picks with large residual may be correct. please increase rsel.]
    # ires: resolution_or_not 
    #       [optional; 1-output resolution file; 0-don’t output. 
    #       Note: only works for the first associated event when the first initiating pick is true. 
    #       Thus, this is only recommended for synthetic resolution analysis]

    ## -S
    np0 = "4"
    ns0 = "4"
    nps0 = "8"
    npsboth0 = "4"
    std0 = "0.5"
    dtps = "0"
    nrt = "1.5"
    drt = "0.5"
    nxd = "0.5"
    rsel = "2"
    
    ##-R
    #-R(rx/rh/tdx/tdh/tint[/gap/GCarc0/latref0/lonref0])(degree/km/degree/km/sec[degree/degree/degree/degree])
    # e.g., 0.1/20/0.02/2/5.0 or 0.1/20/0.02/2/5.0/360 or 0.1/20/0.02/2/5.0/360/180 or 0.1/20/0.02/2/5.0/360/180/42.75/13.25
    # rx: search range in horizontal centered at the station recorded the initiating phase (degree) 
    #       [e.g., > twice of average station interval; smaller than the travel time range]
    # rh: search range in depth (km) 
    #       [e.g., 30 km for crustal earthquakes, will search depth from 0 to 30 km]
    # tdx: search grid size for epicenter (degree)
    # tdh: search grid size for depth (km) 
    #       [time cost mostly depends on the total number of grids! please consider using sparse search grid and strict threshold for quick test. 
    #       Your tdx and tdh don’t have to be the same as that in the travel time table. REAL will automatically interpolate travel time.]
    # tint: two events cannot appear within tint sec, otherwise only keep the most reliable one (tint sec) 
    #       [REAL will use the S travel time within one grid instead if your tint is too small; if not provided, it will use S time window]
    # gap: only keep events with reasonable station gap (e.g., default is 360o)
    # GCarc0: only keep picks with distance of < GCarc0 (e.g., default is 180o) [GCarc0 should be smaller than trx in -G, otherwise it will be replaced by trx-0.05]
    # latref0: reference latitude (degree)
    # lonref0: reference longitude (degree) [latref0 and lonref0 are optional; they will be the search center for all picks. if not provided, the location of the station recording the initiating phases will be used]
    

    ##-R
    rx = "0.2"
    rh = "40"
    tdx = "0.02"
    tdh = "2"
    tint = "0.1"

    D = year + '/' + mon + '/' + day + '/' + '34.475'

    R =  rx + "/" + rh + "/" + tdx + "/" + tdh + "/" + tint

    S = np0 + "/" + ns0 + "/" + nps0 + "/" + npsboth0 + "/" + std0 + "/" + dtps + "/" + nrt + "/" + drt + "/" + nxd + "/" + rsel

    G = "1/40/0.01/1"
    V = "6.2/3.3"

    data_dir = files
    station_file = "./station_for_REAL.dat"
    ttime = "./REAL/tt_db/ttdb_JMA2001.txt"

    result = subprocess.run(['./REAL/REAL', '-D' + D,
                                            '-R' + R,
                                            '-G' + G,
                                            '-S' + S,
                                            '-V' + V,
                                            station_file,
                                            data_dir,
                                            ttime])
    return


if __name__ == '__main__':
    main('./pred_res')
        
    
