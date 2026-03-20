# -*- coding: utf-8 -*-
"""
Lab 3 Collection Script
Radio Interferometry at X-Band

Completed observing script based on:
1. The Lab 3 manual requirements
2. Our Plan for Code for Lab 3
3. The starter code in lab_3_collection.py

Main jobs of this script:
- connect to the interferometer and SNAP
- point to the Sun
- keep updating the telescope pointing during the run
- read one new integrated spectrum per acc_cnt
- save buffered data periodically so nothing is lost
- stow the telescope at the end

IMPORTANT:
Run this on the Raspberry Pi that has ugradio and snap_spec installed.
"""

# ============================================================
# Goal 1 / Goal 2: Imports and setup
# ============================================================

import os
import time
import math
import queue
import threading
from datetime import datetime

import numpy as np
import ugradio  # only works on the Raspberry Pi

# Try both common import styles for snap_spec
try:
    from snap_spec.snap import UGRadioSnap
except Exception:
    try:
        from snap_spec import UGRadioSnap
    except Exception as e:
        raise ImportError(
            "Could not import UGRadioSnap from snap_spec. "
            "Check that snap_spec is installed on the Raspberry Pi."
        ) from e


# ============================================================
# User-configurable observing parameters
# ============================================================

HOST = "localhost"
STREAM_1 = 0
STREAM_2 = 1
SAMPLE_RATE = 500
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

def now_timestamp():
    """Return a readable timestamp for filenames."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


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


def get_sun_altaz(jd=None, lat=LAT_DEG, lon=LON_DEG, alt=ALT_M):
    """
    Get current Sun position in alt/az.
    The lab manual specifies:
    1. ugradio.coord.sunpos(jd) -> ra, dec
    2. ugradio.coord.get_altaz(ra, dec, jd, lat, lon, alt)
    """
    if jd is None:
        jd = ugradio.timing.julian_date()

    ra, dec = ugradio.coord.sunpos(jd)

    # Use the full call form from the lab manual
    alt_deg, az_deg = ugradio.coord.get_altaz(
        ra, dec, jd, lat, lon, alt
    )
    return jd, ra, dec, alt_deg, az_deg


def point_to_sun(ifm, force=False, last_alt=None, last_az=None, last_point_time=None):
    """
    Compute current Sun position and repoint if needed.
    Returns updated pointing info.
    """
    jd, ra, dec, alt_deg, az_deg = get_sun_altaz()

    should_point = force
    if last_alt is None or last_az is None:
        should_point = True
    else:
        moved_deg = angular_sep_deg(last_alt, last_az, alt_deg, az_deg)
        time_since = np.inf if last_point_time is None else (time.time() - last_point_time)

        if moved_deg >= REPOINT_THRESHOLD_DEG or time_since >= MAX_REPOINT_INTERVAL_SEC:
            should_point = True

    if should_point:
        ifm.point(-alt_deg, (az_deg+180)%360)

        # Optional sanity check. get_pointing may return values for both dishes depending on setup.
        # We do not hard-fail if the format is different.
        try:
            current_pointing = ifm.get_pointing()
            print(f"[POINT] Commanded alt={alt_deg:.3f}, az={az_deg:.3f}")
            print(f"[POINT] Reported pointing: {current_pointing}")
        except Exception:
            print(f"[POINT] Commanded alt={alt_deg:.3f}, az={az_deg:.3f}")

        last_alt, last_az = alt_deg, az_deg
        last_point_time = time.time()

    return {
        "jd": jd,
        "ra": ra,
        "dec": dec,
        "alt": alt_deg,
        "az": az_deg,
        "last_alt": last_alt,
        "last_az": last_az,
        "last_point_time": last_point_time,
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
    fname = os.path.join(outdir, f"{prefix}_{now_timestamp()}_{len(records)}rec.npz")

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
# Goal 3 / Goal 4: Interferometer setup and Sun tracking
# ============================================================

def setup_interferometer():
    """
    Create interferometer object, stow first, then point to the Sun.
    """
    ifm = ugradio.interf.Interferometer()

    print("[IFM] Stowing telescope first for safe start.")
    ifm.stow()
    time.sleep(2)

    print("[IFM] Pointing to current Sun position.")
    point_info = point_to_sun(ifm, force=True)

    return ifm, point_info


# ============================================================
# Goal 5 / Goal 6 / Goal 7:
# Read SNAP data, keep acc_cnt unique, and collect data
# ============================================================

def setup_snap():
    """
    Create and initialize the SNAP spectrometer in correlation mode.
    """
    print("[SNAP] Initializing spectrometer.")
    spec = UGRadioSnap(host=HOST, stream_1=STREAM_1, stream_2=STREAM_2)
    spec.initialize(mode=MODE, sample_rate=SAMPLE_RATE)
    return spec


def measure_acc_cnt_timing(spec, n_samples=5):
    """
    Estimate how quickly acc_cnt increments.
    The plan says to determine how fast SNAP is giving data.
    """
    print("[SNAP] Measuring acc_cnt cadence...")
    timings = []
    prev_acc = None

    for i in range(n_samples):
        if prev_acc is None:
            t0 = time.time()
            data = spec.read_data()
            t1 = time.time()
        else:
            t0 = time.time()
            data = spec.read_data(prev_acc)
            t1 = time.time()

        acc = extract_acc_cnt(data)
        timings.append(t1 - t0)
        prev_acc = acc
        print(f"[SNAP] Sample {i+1}/{n_samples}: acc_cnt={acc}, wait={t1-t0:.3f} s")

    if len(timings) > 0:
        print(f"[SNAP] Mean read wait time: {np.mean(timings):.3f} s")
        print(f"[SNAP] Median read wait time: {np.median(timings):.3f} s")

    return timings


# ============================================================
# Goal 8 / Goal 9:
# Background writer thread so collection can continue while saving
# ============================================================

class DataWriter(threading.Thread):
    """
    Background thread that receives buffered records and writes them to disk.
    """
    def __init__(self, outdir, prefix="sun_run"):
        super().__init__(daemon=True)
        self.outdir = outdir
        self.prefix = prefix
        self.q = queue.Queue()
        self.stop_signal = object()
        self.saved_files = []

    def run(self):
        while True:
            item = self.q.get()
            if item is self.stop_signal:
                self.q.task_done()
                break

            try:
                fname = save_records_chunk(item, self.outdir, prefix=self.prefix)
                if fname is not None:
                    self.saved_files.append(fname)
            finally:
                self.q.task_done()

    def submit(self, records):
        self.q.put(records)

    def stop(self):
        self.q.put(self.stop_signal)
        self.q.join()

class PointingThread(threading.Thread):
    def __init__(self, ifm, pointing_queue, stop_event, initial_point):
        super().__init__(daemon=True)
        self.ifm = ifm
        self.q = pointing_queue
        self.stop_event = stop_event
        self.point_info = initial_point

    def run(self):
        while not self.stop_event.is_set():
            self.point_info = point_to_sun(
                self.ifm,
                force=False,
                last_alt=self.point_info["last_alt"],
                last_az=self.point_info["last_az"],
                last_point_time=self.point_info["last_point_time"],
            )

            # Always keep latest pointing (overwrite queue)
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break

            self.q.put(self.point_info)
            time.sleep(0.5)

class DataCollectorThread(threading.Thread):
    def __init__(self, spec, pointing_queue, writer, stop_event):
        super().__init__(daemon=True)
        self.spec = spec
        self.pointing_queue = pointing_queue
        self.writer = writer
        self.stop_event = stop_event

        self.prev_acc_cnt = None
        self.records_buffer = []
        self.n_total = 0
        self.latest_point = None

    def run(self):
        while not self.stop_event.is_set():
            # --- Get latest pointing info (non-blocking) ---
            try:
                self.latest_point = self.pointing_queue.get_nowait()
            except queue.Empty:
                pass  # keep last known

            if self.latest_point is None:
                time.sleep(IDLE_SLEEP_SEC)
                continue

            # --- Read SNAP ---
            if self.prev_acc_cnt is None:
                snap_data = self.spec.read_data()
            else:
                snap_data = self.spec.read_data(self.prev_acc_cnt)

            acc_cnt = extract_acc_cnt(snap_data)

            jd_now = ugradio.timing.julian_date()
            unix_now = time.time()

            record = {
                "jd": jd_now,
                "unix_time": unix_now,
                "ra": self.latest_point["ra"],
                "dec": self.latest_point["dec"],
                "alt": self.latest_point["alt"],
                "az": self.latest_point["az"],
                "acc_cnt": acc_cnt,
                "snap_data": snap_data,
            }

            self.records_buffer.append(record)
            self.n_total += 1

            print(
                f"[DATA] #{self.n_total:05d} "
                f"jd={jd_now:.8f} "
                f"acc_cnt={acc_cnt} "
                f"alt={self.latest_point['alt']:.2f} az={self.latest_point['az']:.2f}"
            )

            self.prev_acc_cnt = acc_cnt

            # --- Send chunk to writer ---
            if len(self.records_buffer) >= SAVE_EVERY_N_RECORDS:
                self.writer.submit(self.records_buffer.copy())
                self.records_buffer.clear()

            time.sleep(IDLE_SLEEP_SEC)

        # flush remaining
        if len(self.records_buffer) > 0:
            self.writer.submit(self.records_buffer.copy())
            self.records_buffer.clear()


# ============================================================
# Main observing function:
# sun_point(time)
# ============================================================

def sun_point(run_hours, outdir="lab3_data", prefix="sun_run", do_timing_check=True):
    """
    Observe the Sun for a specified number of hours.

    Parameters
    ----------
    run_hours : float
        Total observing duration in hours.
    outdir : str
        Directory for saved chunks and final combined file.
    prefix : str
        Filename prefix.
    do_timing_check : bool
        If True, estimate acc_cnt cadence at the start.

    Returns
    -------
    saved_files : list of str
        Chunk files written during the run.
    combined_filename : str
        Final combined file.
    """

    run_seconds = float(run_hours) * 3600.0
    t_start = time.time()
    t_end = t_start + run_seconds

    os.makedirs(outdir, exist_ok=True)

    # ---------- Setup telescope ----------
    ifm, point_info = setup_interferometer()

    # ---------- Setup SNAP ----------
    spec = setup_snap()

    if do_timing_check:
        measure_acc_cnt_timing(spec, n_samples=5)

    # ---------- Setup background writer ----------
    # ---------- Setup threading ----------
    stop_event = threading.Event()

    pointing_queue = queue.Queue(maxsize=1)

    # Start writer (unchanged)
    writer = DataWriter(outdir=outdir, prefix=prefix)
    writer.start()

    # Start pointing thread
    point_thread = PointingThread(ifm, pointing_queue, stop_event, point_info)
    point_thread.start()

    # Start data collector thread
    collector_thread = DataCollectorThread(spec, pointing_queue, writer, stop_event)
    collector_thread.start()

    print(f"[RUN] Running Sun observation for {run_hours:.3f} hours.")
    print(f"[RUN] Output directory: {outdir}")

    try:
        while time.time() < t_end:
            time.sleep(1)

    except KeyboardInterrupt:
        print("[RUN] Interrupted.")

    finally:
        stop_event.set()

        point_thread.join()
        collector_thread.join()

        writer.stop()

        # stow telescope
        try:
            print("[IFM] Stowing telescope.")
            ifm.stow()
        except Exception as e:
            print(f"[WARN] Could not stow telescope: {e}")

        combined_filename = os.path.join(outdir, f"{prefix}_COMBINED_{now_timestamp()}.npz")
        combine_saved_chunks(writer.saved_files, combined_filename)

        print(f"[DONE] Total chunk files: {len(writer.saved_files)}")
        print(f"[DONE] Final combined file: {combined_filename}")

    return writer.saved_files, combined_filename


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    # Example usage:
    # 1. Quick single-spectrum test:
    # data = collect_one_spectrum()

    # 2. Full observing run:
    # Change the hours value to what you need.
    saved_files, combined_file = sun_point(
        run_hours=1.0,
        outdir="lab3_sun_data",
        prefix="sun_observation",
        do_timing_check=False,
    )

    print("[MAIN] Saved chunk files:")
    for fn in saved_files:
        print("   ", fn)

    print("[MAIN] Combined file:")
    print("   ", combined_file)
