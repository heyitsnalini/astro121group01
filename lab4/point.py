# -*- coding: utf-8 -*-
import os
import time
import math
import queue
import threading
from datetime import datetime

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, Galactic, AltAz, EarthLocation
from astropy.coordinates import get_sun

import numpy as np
import ugradio  # only works on the Raspberry Pi


# ============================================================
# User-configurable observing parameters
# ============================================================

AVERAGES_PER_TARGET = 5
SECONDS_PER_AVERAGE = 1
N_SAMPLES_PER_FFT = 1024


SAMPLE_RATE = 2.2e6

# ============================================================
# Calculated observing parameters
# ============================================================

FFTS_PER_SEC = SAMPLE_RATE/N_SAMPLES_PER_FFT

FFTS_PER_AVG = FFTS_PER_SEC * SECONDS_PER_AVERAGE



HOST = "localhost"
MODE = "corr"

# Berkeley interferometer site defaults
# These values are standard approximations for Campbell Hall / Berkeley.
# If your course gives different values, replace them here.
LAT_DEG = 37.87
LON_DEG = -122.26
ALT_M = 200.0

# Repointing logic:
# if the Sun has moved by more than this many degrees, repoint the telescope
REPOINT_THRESHOLD_DEG = 0.25

# Also force a repoint at least this often, even if the threshold is not crossed
MAX_REPOINT_INTERVAL_SEC = 60.0

# How often to flush buffered data to disk
SAVE_EVERY_N_RECORDS = 20

# Small sleep used when polling / waiting
IDLE_SLEEP_SEC = 0.05


# ============================================================
# Utility functions
# ============================================================

def angular_sep_deg(alt1, az1, alt2, az2):
    """
    Approximate angular separation in degrees on the sky.
    Good enough for deciding when to repoint.
    """
    alt1r = math.radians(alt1)
    az1r = math.radians(az1)
    alt2r = math.radians(alt2)
    az2r = math.radians(az2)

    cos_sep = (
        math.sin(alt1r) * math.sin(alt2r)
        + math.cos(alt1r) * math.cos(alt2r) * math.cos(az1r - az2r)
    )
    cos_sep = max(-1.0, min(1.0, cos_sep))
    return math.degrees(math.acos(cos_sep))


def get_altaz(b, l, lat=LAT_DEG, lon=LON_DEG, alt=ALT_M):
    """
    Get alt/az from galactic coord. all should be in degrees.
    """

    gal = SkyCoord(l=l*u.deg, b=b*u.deg, frame=Galactic)

    # 2. Define the Observer Location and Time
    location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=alt*u.m)

    nt = Time.now()

    
    # 3. Create the AltAz Frame
    altaz_frame = AltAz(obstime=nt, location=location)

    # 4. Transform
    altaz_coord = gal.transform_to(altaz_frame)

    alt, az = altaz_coord.alt, altaz_coord.az


    return alt, az, b, l, nt.jd


def point(dish, alt, az, force=False):
    """
    Compute current Sun position and repoint if needed.
    Returns updated pointing info.
    """

    try:
        dish.point(alt, az)
        print(f"[POINT] Commanded alt={alt:.3f}, az={az:.3f}")

    except:
        print(f"[POINT] Pointing failed!!!")


    # Optional sanity check. get_pointing may return values for both dishes depending on setup.
    # We do not hard-fail if the format is different.
    try:
        current_pointing = dish.get_pointing()
        print(f"[POINT] Commanded alt={alt:.3f}, az={az:.3f}")
        print(f"[POINT] Reported pointing: {current_pointing}")
    except Exception:
        print(f"[POINT] Commanded alt={alt:.3f}, az={az:.3f}")


    return {
        "alt": alt,
        "az": az,
    }


def extract_acc_cnt(data):
    """
    Try several common ways of extracting acc_cnt from the SNAP output.
    Returns None if not found.
    """
    # dict-like return
    if isinstance(data, dict):
        for key in ["acc_cnt", "acc_count", "accCnt", "count"]:
            if key in data:
                return data[key]

    # attribute-like return
    for attr in ["acc_cnt", "acc_count", "accCnt", "count"]:
        if hasattr(data, attr):
            return getattr(data, attr)

    return None


def convert_data_to_serializable(data):
    """
    Convert SNAP return data into something that np.savez_compressed can safely store.

    We keep it flexible because the exact return type depends on the installed snap_spec version.
    """
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            try:
                out[k] = np.array(v)
            except Exception:
                out[k] = np.array([str(v)], dtype=object)
        return out

    # For object-like returns, try reading __dict__
    if hasattr(data, "__dict__"):
        out = {}
        for k, v in data.__dict__.items():
            try:
                out[k] = np.array(v)
            except Exception:
                out[k] = np.array([str(v)], dtype=object)
        if out:
            return out

    # Fallback
    return {"raw_data": np.array([data], dtype=object)}


def flatten_record_for_save(record):
    """
    Flatten one record dictionary into a saveable dict.
    """
    base = {
        "jd": np.array(record["jd"]),
        "unix_time": np.array(record["unix_time"]),
        "ra_deg": np.array(record["ra"]),
        "dec_deg": np.array(record["dec"]),
        "alt_deg": np.array(record["alt"]),
        "az_deg": np.array(record["az"]),
    }

    if record["acc_cnt"] is None:
        base["acc_cnt"] = np.array([-1])
    else:
        base["acc_cnt"] = np.array(record["acc_cnt"])

    snap_serial = convert_data_to_serializable(record["snap_data"])
    for k, v in snap_serial.items():
        base[f"snap_{k}"] = v

    return base


def save_records_chunk(records, outdir, prefix="sun_run"):
    """
    Save a list of records to one compressed NPZ file.
    Each chunk is chronologically ordered by construction.
    """
    if len(records) == 0:
        return None

    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{len(records)}rec.npz")

    # Save as object array of dicts because records may not all have identical nested structure
    flattened = [flatten_record_for_save(r) for r in records]
    np.savez_compressed(fname, records=np.array(flattened, dtype=object))

    print(f"[SAVE] Wrote {len(records)} records to {fname}")
    return fname


def combine_saved_chunks(file_list, combined_filename):
    """
    Combine chunk files into one file in time order.
    """
    all_records = []

    for fn in file_list:
        try:
            data = np.load(fn, allow_pickle=True)
            chunk = list(data["records"])
            all_records.extend(chunk)
        except Exception as e:
            print(f"[WARN] Could not read {fn}: {e}")

    # Sort by JD if possible
    def get_jd(rec):
        try:
            return float(rec["jd"])
        except Exception:
            return np.inf

    all_records.sort(key=get_jd)
    np.savez_compressed(combined_filename, records=np.array(all_records, dtype=object))
    print(f"[FINAL SAVE] Combined {len(all_records)} total records into {combined_filename}")




# ============================================================
# Main observing function:
# ============================================================

def get_data(targets, time_limit=12, outdir="lab4_data", prefix="uhh"):
    run_seconds = float(time_limit) * 3600.0
    t_start = time.time()
    t_end = t_start + run_seconds

    os.makedirs(outdir, exist_ok=True)

    # ---------- Setup telescope ----------
    dish = ugradio.leusch.LeuschTelescope()

    # print("[DISH] Stowing telescope first for safe start.")
    # dish.stow()
    # time.sleep(2)

    # print("[DISH] Maintenance position to dump water.")
    # dish.maintenance()
    # time.sleep(2)

    # ---------- Setup SDRs ----------

    print("[SDR] Initializing SDRs.")
    sdr0 = ugradio.sdr.SDR(device_index=0, direct=False, sample_rate=SAMPLE_RATE)
    sdr1 = ugradio.sdr.SDR(device_index=1, direct=False, sample_rate=SAMPLE_RATE)

    print(f"[RUN] Running observation for {time_limit:.3f} hours.")
    print(f"[RUN] Output directory: {outdir}")

    try:
        for target in targets:
            l, b = target
            alt, az, b_, l_, jd = get_altaz(b, l)
            point(dish, alt=50 , az=50)

            target_outputs = {}

            for i in np.arange(AVERAGES_PER_TARGET):
                print('Average ', i)
                try:
                    # Capture data - the capture_data function handles its own event loop
                    data0 = sdr0.capture_data(nsamples=N_SAMPLES_PER_FFT, nblocks=int(1+FFTS_PER_AVG))
                    data1 = sdr1.capture_data(nsamples=N_SAMPLES_PER_FFT, nblocks=int(1+FFTS_PER_AVG))

                    output = []
                    for data in [data0, data1]:
                        avg = []
                        for block in data:
                            data_f = np.fft.fft(block)
                            freq = np.fft.fftfreq(len(block), d=1/SAMPLE_RATE)
                            
                            data_f = np.fft.fftshift(data_f)
                            freq = np.fft.fftshift(freq)
                            
                            avg.append(data_f)
                        
                        avg = np.mean(avg, axis=0)
                        output.append(avg)
                        print("[RUN] Saved AVERAGE with shape ", avg.shape)
                    
                    target_outputs[i] = output
                    
                except Exception as e:
                    print(f"[ERROR] Capture failed on average {i}: {e}")
                    continue
                
            # Save after each target
            np.savez(f'{prefix}-{SAMPLE_RATE/1e6}MHz', data=target_outputs, l=l, b=b, time=jd, sample_rate=SAMPLE_RATE)
            print(f'Collecting at {SAMPLE_RATE/1e6} MHz')

    except KeyboardInterrupt:
        print("[RUN] Interrupted.")

    finally:
        # Shutdown SDRs properly
        try:
            print("[SDR] Stopping SDRs.")
            # Let the event loop in capture_data close naturally
            # Just close the SDR connections
            sdr0.close()
            sdr1.close()
            time.sleep(0.5)  # Give threads time to die
        except Exception as e:
            print(f"[WARN] Error closing SDRs: {e}")

        try:
            print("[DISH] Stowing telescope.")
            dish.stow()
        except Exception as e:
            print(f"[WARN] Could not stow telescope: {e}")

        # combined_filename = os.path.join(outdir, f"{prefix}_COMBINED_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.npz")
        # combine_saved_chunks(writer.saved_files, combined_filename)

        # print(f"[DONE] Total chunk files: {len(writer.saved_files)}")
        # print(f"[DONE] Final combined file: {combined_filename}")

    # return writer.saved_files, combined_filename



# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    # Example usage:
    # 1. Quick single-spectrum test:
    # data = collect_one_spectrum()

    # 2. Full observing run:
    # Change the hours value to what you need.
    get_data(
        targets=[(145, -20)],
        outdir="lab4_script_test",
        prefix="test",
    )

    # print("[MAIN] Saved chunk files:")
    # for fn in saved_files:
    #     print("   ", fn)

    # print("[MAIN] Combined file:")
    # print("   ", combined_file)
