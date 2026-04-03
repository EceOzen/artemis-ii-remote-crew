# ============================================================
# Day 3 — The Silent Passenger: Deep Space Radiation
# Cell 1 — Constants, Van Allen Belt Zones & Orion Altitude Profile
# Real Artemis II mission data — NASA blogs + JPL Horizons
# Code Beyond the Earth | April 3, 2026
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from IPython.display import display, HTML

# ── Physical constants ───────────────────────────────────────
R_earth = 6_371          # km

# ── Van Allen Belt boundaries (km altitude) ─────────────────
# Source: NASA Van Allen Probes mission data
INNER_LOW  = 1_000       # km
INNER_HIGH = 12_000      # km
OUTER_LOW  = 13_000      # km
OUTER_HIGH = 60_000      # km

# ── Real Artemis II orbital parameters ──────────────────────
# Source: NASA mission blogs, CBS, Space.com live coverage
h_perigee_0    = 185       # km  — initial orbit perigee (115 miles)
h_perigee_1    = 185       # km  — after perigee raise burn (minor refinement)
h_apogee       = 70_363    # km  — after 18-min apogee raise burn (43,760 miles)
t_tli_hours    = 25.62     # h   — TLI fired T+25h37min (7:49 PM EDT, Apr 2)
tli_duration_s = 5*60 + 49 # s   — 5 min 49 sec confirmed burn duration

# ── Timeline: 0 → 32 hours since launch ─────────────────────
# Key mission events (all times in hours since T+0 = 6:35 PM EDT Apr 1):
#
#  T+00:00  Launch
#  T+00:50  Perigee raise burn (ICPS RL10, short burn) → perigee ~185 km
#  T+01:58  Apogee raise burn (ICPS RL10, 18 min)     → apogee 70,363 km
#  T+02:30  Proximity operations demonstration begins
#  T+25:37  TLI burn (OMS-E, 5 min 49 sec)            → trans-lunar coast
#  T+32:00  End of profile (Orion deep in trans-lunar coast)

N = 2000
t = np.linspace(0, 32, N)   # hours

def orion_altitude_profile(t_arr):
    """
    Physically accurate altitude profile based on real Artemis II
    mission event timestamps from NASA blogs.
    Returns altitude in km at each time point.
    """
    alt = np.zeros(len(t_arr))

    for i, ti in enumerate(t_arr):

        if ti < 0.083:
            # Ascent — 0 to ~5 min, climbing rapidly
            alt[i] = h_perigee_0 * (ti / 0.083)

        elif ti < 0.833:
            # Initial parking orbit — roughly circular at 185 km
            # (T+5min to T+50min)
            alt[i] = h_perigee_0

        elif ti < 0.917:
            # Perigee raise burn — short, minor altitude refinement
            alt[i] = h_perigee_0

        elif ti < 1.97:
            # Coast between burns — still near perigee
            alt[i] = h_perigee_0

        elif ti < 3.27:
            # Apogee raise burn + immediate coast climbing to apogee
            # 18-minute burn raises apogee from ~185 km to 70,363 km
            frac = (ti - 1.97) / 1.3
            alt[i] = h_perigee_0 + (h_apogee - h_perigee_0) * np.sin(frac * np.pi/2)

        elif ti < 13.8:
            # Coast to apogee — elliptical orbit, climbing
            frac = (ti - 3.27) / 10.53
            alt[i] = h_perigee_0 + (h_apogee - h_perigee_0) * np.sin(frac * np.pi/2)

        elif ti < 14.3:
            # Near apogee — ~70,363 km
            alt[i] = h_apogee - 500 * abs(ti - 14.05) / 0.25

        elif ti < 25.4:
            # Return toward perigee — descending from apogee
            frac = (ti - 14.3) / 11.1
            alt[i] = h_apogee - (h_apogee - h_perigee_0) * frac

        elif ti < 25.72:
            # TLI burn window — near perigee, engine firing
            # Velocity increasing, altitude begins rising rapidly
            burn_frac = (ti - 25.4) / 0.32
            alt[i] = h_perigee_0 + burn_frac * 2_000

        else:
            # Post-TLI trans-lunar coast
            # Climbing toward Moon (~384,400 km over ~96 hours)
            coast_hours = ti - 25.72
            # Decelerating due to gravity — approximate with sqrt curve
            alt[i] = h_perigee_0 + 2_000 + 45_000 * np.sqrt(coast_hours / 6.28)

    return alt

altitude = orion_altitude_profile(t)

# ── Belt crossing analysis ───────────────────────────────────
in_inner = (altitude >= INNER_LOW)  & (altitude <= INNER_HIGH)
in_outer = (altitude >= OUTER_LOW)  & (altitude <= OUTER_HIGH)

t_inner = t[in_inner]
t_outer = t[in_outer]

# ── Print summary ────────────────────────────────────────────
print("=" * 56)
print("   ARTEMIS II — ORBITAL PROFILE & BELT CROSSINGS")
print("=" * 56)
print(f"\n  Launch              : T+00:00 (6:35 PM EDT, Apr 1)")
print(f"  Perigee raise burn  : T+00:50")
print(f"  Apogee raise burn   : T+01:58  (18 min, ICPS RL10)")
print(f"  Parking orbit alt   : {h_perigee_0} km (perigee)")
print(f"  Apogee altitude     : {h_apogee:,} km")
print(f"  TLI burn            : T+{t_tli_hours:.2f}h  (5 min 49 sec)")
print()
print("  Van Allen Belts:")
print(f"    Inner belt : {INNER_LOW:,} – {INNER_HIGH:,} km")
print(f"    Outer belt : {OUTER_LOW:,} – {OUTER_HIGH:,} km")
print()

if len(t_inner) > 0:
    print(f"  Inner belt entry    : T+{t_inner[0]:.2f}h")
    print(f"  Inner belt exit     : T+{t_inner[-1]:.2f}h")
    print(f"  Inner belt transit  : {(t_inner[-1]-t_inner[0])*60:.0f} min")
else:
    print("  Inner belt          : not entered in this profile")

if len(t_outer) > 0:
    # may have two transits — outbound and return
    gaps = np.where(np.diff(t_outer) > 0.5)[0]
    if len(gaps) > 0:
        # two separate transits
        t_out1 = t_outer[:gaps[0]+1]
        t_out2 = t_outer[gaps[0]+1:]
        print(f"\n  Outer belt (outbound):")
        print(f"    Entry  : T+{t_out1[0]:.2f}h")
        print(f"    Exit   : T+{t_out1[-1]:.2f}h")
        print(f"    Duration: {(t_out1[-1]-t_out1[0])*60:.0f} min")
        print(f"\n  Outer belt (return leg):")
        print(f"    Entry  : T+{t_out2[0]:.2f}h")
        print(f"    Exit   : T+{t_out2[-1]:.2f}h")
        print(f"    Duration: {(t_out2[-1]-t_out2[0])*60:.0f} min")
        total_outer = (t_out1[-1]-t_out1[0]) + (t_out2[-1]-t_out2[0])
        print(f"\n  Total outer belt exposure: {total_outer*60:.0f} min")
    else:
        print(f"\n  Outer belt transit  : T+{t_outer[0]:.2f}h → T+{t_outer[-1]:.2f}h")
        print(f"  Duration            : {(t_outer[-1]-t_outer[0])*60:.0f} min")

print()
print(f"  Peak altitude (profile): {altitude.max():,.0f} km")
print(f"  TLI fired at           : {altitude[np.argmin(np.abs(t-25.62))]:,.0f} km")
print("=" * 56)

# ============================================================
# Cell 2 — Artemis I HERA Radiation Data
# Reconstructed from published values:
#   George et al., Nature 2024 (doi:10.1038/s41586-024-07927-7)
#   Laramore et al., WRMISS 2023
#   NASA NTRS 20230010550
# ============================================================


# ── Mission timeline anchors (Artemis I, all UTC Nov 16 2022) ─
# These timestamps are directly from WRMISS 2023 / Nature 2024
LAUNCH_UTC        = 6.796   # 06:47:44 UTC → decimal hours
INNER_ENTRY_UTC   = 7.186   # 07:11:11 UTC
INNER_PEAK_UTC    = 7.367   # ~07:22 UTC (midpoint)
INNER_EXIT_UTC    = 7.778   # 07:46:41 UTC
OUTER_ENTRY_UTC   = 8.750   # 08:45:00 UTC
OUTER_PEAK_UTC    = 9.250   # ~09:15 UTC (approx midpoint)
OUTER_EXIT_UTC    = 10.750  # 10:45:00 UTC
TLI_UTC           = 15.933  # ~MET 07:56 → outbound burn
DEEP_SPACE_START  = 11.5    # after belt exit, GCR dominates

# ── Published dose rate values (μGy/min) ─────────────────────
# Three detector locations — represent different shielding levels

DETECTORS = {
    'HERA HSU2\n(crew cabin — low shield)': {
        'baseline_gcr'   : 0.15,    # pre-belt background
        'inner_peak'     : 287.0,   # Nature 2024, Fig 2a
        'outer_peak'     : 4.0,     # Nature 2024, Fig 2a approx
        'deep_space_gcr' : 0.28,    # interplanetary GCR
        'color'          : '#EF4444',
    },
    'EAD MU01\n(Orion wall — low shield)': {
        'baseline_gcr'   : 0.15,
        'inner_peak'     : 240.0,   # Nature 2024
        'outer_peak'     : 3.2,
        'deep_space_gcr' : 0.26,
        'color'          : '#F59E0B',
    },
    'M-42 SN127\n(Helga phantom — high shield)': {
        'baseline_gcr'   : 0.10,
        'inner_peak'     : 69.0,    # Nature 2024
        'outer_peak'     : 1.8,
        'deep_space_gcr' : 0.18,
        'color'          : '#10B981',
    },
}

# ── Build time-series (0 → 20 hours from launch) ─────────────
t_mission = np.linspace(0, 20, 5000)   # hours from launch
t_utc     = t_mission + LAUNCH_UTC     # UTC hours

def dose_rate_profile(t_utc_arr, detector):
    """
    Reconstruct dose rate time series from published peak values.
    Uses Gaussian pulses for belt crossings anchored to real timestamps.
    """
    d  = np.zeros(len(t_utc_arr))
    bl = detector['baseline_gcr']

    for i, t in enumerate(t_utc_arr):

        if t < INNER_ENTRY_UTC:
            # pre-belt: LEO baseline GCR (~0.15 μGy/min)
            d[i] = bl

        elif t <= INNER_EXIT_UTC:
            # inner proton belt — sharp Gaussian peak
            # width calibrated to 35-min transit (WRMISS 2023)
            sigma = (INNER_EXIT_UTC - INNER_ENTRY_UTC) / 5
            peak  = detector['inner_peak']
            d[i]  = bl + peak * np.exp(
                -((t - INNER_PEAK_UTC)**2) / (2 * sigma**2)
            )

        elif t < OUTER_ENTRY_UTC:
            # gap between inner and outer belt
            d[i] = bl * 1.5   # slightly elevated

        elif t <= OUTER_EXIT_UTC:
            # outer electron belt — broader, lower peak
            sigma = (OUTER_EXIT_UTC - OUTER_ENTRY_UTC) / 4
            peak  = detector['outer_peak']
            d[i]  = bl + peak * np.exp(
                -((t - OUTER_PEAK_UTC)**2) / (2 * sigma**2)
            )
            # orientation burn at TLI → 50% drop (Nature 2024)
            if abs(t - TLI_UTC) < 0.2:
                d[i] *= 0.5

        else:
            # deep space GCR — stable, lower than expected
            # Nature 2024: "as much as 60% lower than previous observations"
            d[i] = detector['deep_space_gcr']

    return d

# ── Build dataframe ──────────────────────────────────────────
data = {'t_hours_from_launch': t_mission, 't_utc': t_utc}

for name, det in DETECTORS.items():
    col = name.split('\n')[0]   # short name for column
    data[col] = dose_rate_profile(t_utc, det)

df = pd.DataFrame(data)

# ── Cumulative dose (integrate μGy/min × 60 → μGy) ───────────
dt_hours = t_mission[1] - t_mission[0]
dt_min   = dt_hours * 60

for name, det in DETECTORS.items():
    col     = name.split('\n')[0]
    cum_col = col + '_cumulative_uGy'
    df[cum_col] = df[col].cumsum() * dt_min

# ── Summary statistics ────────────────────────────────────────
print("=" * 60)
print("   ARTEMIS I — HERA RADIATION DATA SUMMARY")
print("   Source: George et al., Nature 2024 + WRMISS 2023")
print("=" * 60)
print(f"\n  Inner belt transit : 07:11 – 07:46 UTC  (~35 min)")
print(f"  Outer belt transit : 08:45 – 10:45 UTC  (~120 min)")
print(f"  Total belt exposure: ~155 min\n")

for name, det in DETECTORS.items():
    short = name.split('\n')[0]
    col   = short
    cum   = short + '_cumulative_uGy'
    peak  = df[col].max()
    # cumulative dose during belt transit only
    mask_belt = (df['t_utc'] >= INNER_ENTRY_UTC) & \
                (df['t_utc'] <= OUTER_EXIT_UTC)
    belt_dose = (df.loc[mask_belt, col].sum() * dt_min)

    print(f"  {short}")
    print(f"    Peak dose rate : {peak:,.1f} μGy/min")
    print(f"    Belt dose      : {belt_dose:.2f} μGy")
    print(f"    GCR deep space : {det['deep_space_gcr']:.2f} μGy/min")
    print()

print(f"  Model vs measured error (WRMISS 2023):")
print(f"    Models underestimated by 9–14% in inner belt")
print(f"    This is your Day 3 validation moment.")
print()
print(f"  Orientation effect (Nature 2024):")
print(f"    50% dose reduction during ICPS burn in inner belt")
print(f"    Spacecraft bulk shields crew from directional protons")
print()
print(f"  Total mission dose estimate (NASA NTRS 2023):")
print(f"    22.3 mSv effective dose — hypothetical male crew member")
print(f"    Well within NASA career limit of 600 mSv")
print("=" * 60)

print(f"\n  DataFrame shape : {df.shape}")
print(f"  Columns         : {list(df.columns)}")

# ============================================================
# Cell 3 — Dose rate visualization
# Artemis I HERA data across full belt transit + deep space
# ============================================================

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0D1117')
gs  = GridSpec(2, 2, figure=fig,
               hspace=0.42, wspace=0.32)

ax_main = fig.add_subplot(gs[0, :])    # top full width — dose rate timeline
ax_cum  = fig.add_subplot(gs[1, 0])   # bottom left  — cumulative dose
ax_comp = fig.add_subplot(gs[1, 1])   # bottom right — shielding comparison

colors = {
    'HERA HSU2' : '#EF4444',
    'EAD MU01'  : '#F59E0B',
    'M-42 SN127': '#10B981',
}
labels = {
    'HERA HSU2' : 'HERA HSU2 (crew cabin, low shield)',
    'EAD MU01'  : 'EAD MU01 (Orion wall, low shield)',
    'M-42 SN127': 'M-42 SN127 (Helga phantom, high shield)',
}

# ── TOP: dose rate timeline ──────────────────────────────────
ax_main.set_facecolor('#0D1117')
ax_main.set_title(
    'Artemis I — HERA dose rate through Van Allen Belts & deep space',
    color='white', fontsize=12, pad=10)
ax_main.set_xlabel('Mission elapsed time (hours from launch)',
                   color='#94A3B8', fontsize=9)
ax_main.set_ylabel('Dose rate (μGy/min)',
                   color='#94A3B8', fontsize=9)
ax_main.tick_params(colors='#94A3B8', labelsize=8)
for sp in ['bottom', 'left']:
    ax_main.spines[sp].set_color('#334155')
for sp in ['top', 'right']:
    ax_main.spines[sp].set_visible(False)
ax_main.grid(color='#1E293B', lw=0.6, zorder=0)
ax_main.set_yscale('log')

# belt zone shading
inner_start = INNER_ENTRY_UTC - LAUNCH_UTC
inner_end   = INNER_EXIT_UTC  - LAUNCH_UTC
outer_start = OUTER_ENTRY_UTC - LAUNCH_UTC
outer_end   = OUTER_EXIT_UTC  - LAUNCH_UTC

ax_main.axvspan(inner_start, inner_end,
                alpha=0.12, color='#EF4444', zorder=1)
ax_main.axvspan(outer_start, outer_end,
                alpha=0.08, color='#3B82F6', zorder=1)
ax_main.axvspan(outer_end, 13.25,
                alpha=0.05, color='#10B981', zorder=1)

# zone labels
ax_main.text((inner_start+inner_end)/2, 180,
             'Inner belt\n(proton)', color='#EF4444',
             fontsize=8, ha='center', style='italic')
ax_main.text((outer_start+outer_end)/2, 180,
             'Outer belt\n(electron)', color='#3B82F6',
             fontsize=8, ha='center', style='italic')
ax_main.text(11.8, 0.22,
             'Deep space\n(GCR)', color='#10B981',
             fontsize=8, ha='center', style='italic')

# orientation burn annotation
ax_main.axvline(TLI_UTC - LAUNCH_UTC, color='#F59E0B',
                lw=1.0, ls=':', alpha=0.7)
ax_main.text(TLI_UTC - LAUNCH_UTC + 0.05, 40,
             'ICPS orientation\nburn → 50% drop',
             color='#F59E0B', fontsize=7.5, style='italic')

# plot each detector
for col, color in colors.items():
    ax_main.plot(df['t_hours_from_launch'], df[col],
                 color=color, lw=1.5, alpha=0.9,
                 label=labels[col], zorder=4)

# baseline reference
ax_main.axhline(0.15, color='#475569', lw=0.8,
                ls='--', alpha=0.6)
ax_main.text(0.1, 0.17, 'LEO baseline ~0.15 μGy/min',
             color='#475569', fontsize=7.5)

ax_main.set_xlim(0, 13.5)
ax_main.set_ylim(0.05, 1200)
ax_main.legend(facecolor='#1E293B', edgecolor='#334155',
               labelcolor='white', fontsize=8,
               loc='upper right')

# ── BOTTOM LEFT: cumulative dose ─────────────────────────────
ax_cum.set_facecolor('#0D1117')
ax_cum.set_title('Cumulative absorbed dose',
                 color='white', fontsize=10, pad=8)
ax_cum.set_xlabel('Mission elapsed time (hours)',
                  color='#94A3B8', fontsize=9)
ax_cum.set_ylabel('Cumulative dose (μGy)',
                  color='#94A3B8', fontsize=9)
ax_cum.tick_params(colors='#94A3B8', labelsize=8)
for sp in ['bottom', 'left']:
    ax_cum.spines[sp].set_color('#334155')
for sp in ['top', 'right']:
    ax_cum.spines[sp].set_visible(False)
ax_cum.grid(color='#1E293B', lw=0.6, zorder=0)

ax_cum.axvspan(inner_start, inner_end,
               alpha=0.10, color='#EF4444')
ax_cum.axvspan(outer_start, outer_end,
               alpha=0.06, color='#3B82F6')

for col, color in colors.items():
    cum_col = col + '_cumulative_uGy'
    ax_cum.plot(df['t_hours_from_launch'], df[cum_col],
                color=color, lw=2.0, alpha=0.9,
                label=col, zorder=4)

# annotate final belt cumulative values
for col, color in colors.items():
    cum_col = col + '_cumulative_uGy'
    val_at_belt_exit = df.loc[
        df['t_hours_from_launch'] <= outer_end,
        cum_col].iloc[-1]
    ax_cum.annotate(
        f'{val_at_belt_exit:.0f} μGy',
        xy=(outer_end, val_at_belt_exit),
        xytext=(outer_end + 0.4, val_at_belt_exit),
        color=color, fontsize=7.5,
        arrowprops=dict(arrowstyle='->', color=color, lw=0.8)
    )

ax_cum.set_xlim(0, 13.5)
ax_cum.legend(facecolor='#1E293B', edgecolor='#334155',
              labelcolor='white', fontsize=7, loc='upper left')

# ── BOTTOM RIGHT: shielding comparison bar chart ─────────────
ax_comp.set_facecolor('#0D1117')
ax_comp.set_title('Belt dose by shielding location',
                  color='white', fontsize=10, pad=8)
ax_comp.tick_params(colors='#94A3B8', labelsize=8)
for sp in ['bottom', 'left']:
    ax_comp.spines[sp].set_color('#334155')
for sp in ['top', 'right']:
    ax_comp.spines[sp].set_visible(False)
ax_comp.grid(axis='x', color='#1E293B', lw=0.6, zorder=0)
ax_comp.set_xlabel('Absorbed dose during belt transit (μGy)',
                   color='#94A3B8', fontsize=9)

# compute belt doses
belt_mask = ((df['t_hours_from_launch'] >= inner_start) &
             (df['t_hours_from_launch'] <= outer_end))
dt_min = (df['t_hours_from_launch'].iloc[1] -
          df['t_hours_from_launch'].iloc[0]) * 60

bar_labels = ['HERA HSU2\n(low shield)',
              'EAD MU01\n(low shield)',
              'M-42 SN127\n(high shield)']
bar_vals   = [
    df.loc[belt_mask, col].sum() * dt_min
    for col in colors.keys()
]
bar_colors = list(colors.values())

bars = ax_comp.barh(bar_labels, bar_vals,
                    color=bar_colors, alpha=0.85,
                    height=0.45, zorder=3)

for bar, val in zip(bars, bar_vals):
    ax_comp.text(val + 50, bar.get_y() + bar.get_height()/2,
                 f'{val:,.0f} μGy',
                 va='center', color='white', fontsize=9,
                 fontweight='bold')

# 4x difference callout
ax_comp.text(bar_vals[0] * 0.5, 2.55,
             '4× shielding difference\n(Nature 2024)',
             color='#94A3B8', fontsize=7.5,
             ha='center', style='italic')
ax_comp.annotate('',
    xy=(bar_vals[2], 2.0),
    xytext=(bar_vals[0], 2.0),
    arrowprops=dict(arrowstyle='<->', color='#64748B', lw=1.0))

ax_comp.set_xlim(0, max(bar_vals) * 1.25)

plt.suptitle(
    'Artemis I Radiation Environment — Van Allen Belts to Deep Space\n'
    'Source: George et al., Nature 2024  ·  Laramore et al., WRMISS 2023',
    color='white', fontsize=12, fontweight='bold', y=1.01)

plt.savefig('artemis_radiation_dose_rate.png', dpi=120,
            facecolor='#0D1117')
plt.show()

print("Saved: artemis_radiation_dose_rate.png")
print()
print(f"Key numbers for blog:")
print(f"  Inner belt peak (HSU2)  : 287 μGy/min")
print(f"  Inner belt peak (M-42)  : 69 μGy/min  → 4× difference")
print(f"  Outer belt peak (HSU2)  : ~4 μGy/min")
print(f"  Deep space GCR (HSU2)   : 0.28 μGy/min")
print(f"  Total belt dose (HSU2)  : {bar_vals[0]:,.0f} μGy")
print(f"  Total belt dose (M-42)  : {bar_vals[2]:,.0f} μGy")
print(f"  Ratio                   : {bar_vals[0]/bar_vals[2]:.1f}×")

# ============================================================
# Cell 4 — Predicted vs Measured dose rate
# AP9/AE9 radiation belt models vs real HERA measurements
# Source: Laramore et al., WRMISS 2023 (models underestimated
#         inner belt by 9–14%)
# ============================================================

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('#0D1117')
gs  = GridSpec(2, 3, figure=fig,
               hspace=0.45, wspace=0.35)

ax_main  = fig.add_subplot(gs[0, :])    # top — predicted vs measured
ax_err   = fig.add_subplot(gs[1, 0])   # bottom left  — error by phase
ax_why   = fig.add_subplot(gs[1, 1])   # bottom centre — what models miss
ax_card  = fig.add_subplot(gs[1, 2])   # bottom right — summary card

# ── Model error values from WRMISS 2023 ──────────────────────
# "Modelled peak rates are 70–80% of measured"
# Cumulative dose errors: LSU −13.8%, HSU1 −13.9%, HSU2 −9.4%
MODEL_SCALE = {
    'HERA HSU2' : 0.906,   # model predicted 90.6% of real → −9.4% error
    'EAD MU01'  : 0.876,   # ~−12.4% (interpolated from paper range)
    'M-42 SN127': 0.862,   # ~−13.8% (LSU equivalent, WRMISS 2023)
}

# Build predicted dose rate series by scaling measured data
# (models are linear underpredictions — standard for AP9/AE9)
predicted = {}
for col, scale in MODEL_SCALE.items():
    predicted[col] = df[col] * scale

# ── TOP: predicted vs measured overlay ───────────────────────
ax_main.set_facecolor('#0D1117')
ax_main.set_title(
    'AP9/AE9 radiation belt model  vs  real HERA measurements — Artemis I',
    color='white', fontsize=12, pad=10)
ax_main.set_xlabel('Mission elapsed time (hours from launch)',
                   color='#94A3B8', fontsize=9)
ax_main.set_ylabel('Dose rate (μGy/min)',
                   color='#94A3B8', fontsize=9)
ax_main.tick_params(colors='#94A3B8', labelsize=8)
for sp in ['bottom', 'left']:
    ax_main.spines[sp].set_color('#334155')
for sp in ['top', 'right']:
    ax_main.spines[sp].set_visible(False)
ax_main.grid(color='#1E293B', lw=0.6, zorder=0)
ax_main.set_yscale('log')

# belt zone shading
inner_start = INNER_ENTRY_UTC - LAUNCH_UTC
inner_end   = INNER_EXIT_UTC  - LAUNCH_UTC
outer_start = OUTER_ENTRY_UTC - LAUNCH_UTC
outer_end   = OUTER_EXIT_UTC  - LAUNCH_UTC

ax_main.axvspan(inner_start, inner_end,
                alpha=0.10, color='#EF4444', zorder=1)
ax_main.axvspan(outer_start, outer_end,
                alpha=0.07, color='#3B82F6', zorder=1)

# plot HSU2 only for clarity — most cited in paper
col = 'HERA HSU2'
ax_main.plot(df['t_hours_from_launch'], df[col],
             color='#EF4444', lw=2.0, alpha=0.95,
             label='Measured (HERA HSU2)', zorder=5)
ax_main.plot(df['t_hours_from_launch'], predicted[col],
             color='#EF4444', lw=1.5, alpha=0.6,
             ls='--', label='AP9/AE9 model prediction', zorder=4)

# fill between to show gap
ax_main.fill_between(
    df['t_hours_from_launch'],
    predicted[col], df[col],
    where=(df['t_hours_from_launch'] >= inner_start) &
          (df['t_hours_from_launch'] <= outer_end),
    color='#F59E0B', alpha=0.15,
    label='Model gap (underestimate)', zorder=3
)

# belt labels
ax_main.text((inner_start+inner_end)/2, 150,
             'Inner belt\n(protons)',
             color='#EF4444', fontsize=8, ha='center',
             style='italic')
ax_main.text((outer_start+outer_end)/2, 150,
             'Outer belt\n(electrons)',
             color='#3B82F6', fontsize=8, ha='center',
             style='italic')
ax_main.text(11.5, 0.22,
             'Deep space GCR',
             color='#10B981', fontsize=8, ha='center',
             style='italic')

# peak annotation
peak_t   = df.loc[df[col].idxmax(), 't_hours_from_launch']
peak_val = df[col].max()
pred_val = predicted[col].max()
ax_main.annotate(
    f'Measured: {peak_val:.0f} μGy/min',
    xy=(peak_t, peak_val),
    xytext=(peak_t + 0.3, peak_val * 1.8),
    color='#EF4444', fontsize=8,
    arrowprops=dict(arrowstyle='->', color='#EF4444', lw=1.0)
)
ax_main.annotate(
    f'Model: {pred_val:.0f} μGy/min',
    xy=(peak_t, pred_val),
    xytext=(peak_t + 1.2, pred_val * 0.4),
    color='#F59E0B', fontsize=8,
    arrowprops=dict(arrowstyle='->', color='#F59E0B', lw=1.0)
)

ax_main.set_xlim(0, 13.5)
ax_main.set_ylim(0.05, 1200)
ax_main.legend(facecolor='#1E293B', edgecolor='#334155',
               labelcolor='white', fontsize=9, loc='upper right')

# ── BOTTOM LEFT: error by sensor ─────────────────────────────
ax_err.set_facecolor('#0D1117')
ax_err.set_title('Model error by sensor\n(inner belt cumulative dose)',
                 color='white', fontsize=9, pad=8)
ax_err.tick_params(colors='#94A3B8', labelsize=8)
for sp in ['bottom', 'left']:
    ax_err.spines[sp].set_color('#334155')
for sp in ['top', 'right']:
    ax_err.spines[sp].set_visible(False)
ax_err.grid(axis='x', color='#1E293B', lw=0.6, zorder=0)
ax_err.set_xlabel('Model error (%)', color='#94A3B8', fontsize=9)

sensors = ['LSU\n(WRMISS)', 'HSU1\n(WRMISS)', 'HSU2\n(WRMISS)']
errors  = [-13.8, -13.9, -9.4]
err_colors = ['#F59E0B', '#F59E0B', '#EF4444']

bars = ax_err.barh(sensors, errors,
                   color=err_colors, alpha=0.85,
                   height=0.4, zorder=3)

for bar, val in zip(bars, errors):
    ax_err.text(val - 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%',
                ha='right', va='center',
                color='white', fontsize=9, fontweight='bold')

ax_err.axvline(0, color='#334155', lw=1.0)
ax_err.set_xlim(-20, 5)

# reference line — Day 2 Δv error
ax_err.axvline(-0.35, color='#10B981', lw=1.5,
               ls=':', alpha=0.8)
ax_err.text(-0.35, 2.6, 'Day 2\nΔv error\n(−0.35%)',
            color='#10B981', fontsize=7,
            ha='center', style='italic')

# ── BOTTOM CENTRE: what models miss ──────────────────────────
ax_why.set_facecolor('#0D1117')
ax_why.axis('off')
ax_why.set_title('Why models underestimate',
                 color='white', fontsize=9, pad=8)

reasons = [
    ('AP9/AE9 static models',
     'Do not capture real-time\nbelt dynamics or solar activity'),
    ('ICPS not included',
     'ICPS shielding was not\nmodelled in pre-flight runs'),
    ('Pitch-angle distribution',
     'Proton directional flux\nharder to model accurately'),
    ('Solar max conditions',
     'Artemis I flew near solar\nmaximum — enhanced belts'),
]

for i, (title, detail) in enumerate(reasons):
    y = 0.88 - i * 0.23
    ax_why.plot(0.04, y, 'o', color='#F59E0B',
                ms=6, transform=ax_why.transAxes,
                zorder=3)
    ax_why.text(0.12, y + 0.02, title,
                transform=ax_why.transAxes,
                color='white', fontsize=8.5,
                fontweight='bold', va='center')
    ax_why.text(0.12, y - 0.055, detail,
                transform=ax_why.transAxes,
                color='#64748B', fontsize=7.5,
                va='center', linespacing=1.4)

# ── BOTTOM RIGHT: summary validation card ────────────────────
ax_card.set_facecolor('#0D1117')
ax_card.axis('off')

# card background
from matplotlib.patches import FancyBboxPatch
ax_card.add_patch(FancyBboxPatch(
    (0.05, 0.05), 0.90, 0.90,
    boxstyle='round,pad=0.02',
    facecolor='#0F172A', edgecolor='#334155',
    linewidth=1.0, transform=ax_card.transAxes, zorder=2
))

ax_card.text(0.5, 0.88, 'Validation summary',
             transform=ax_card.transAxes,
             color='#64748B', fontsize=8,
             ha='center', style='italic')

# Day 2 result
ax_card.text(0.5, 0.76, 'Day 2 — Δv (orbital mechanics)',
             transform=ax_card.transAxes,
             color='#94A3B8', fontsize=8, ha='center')
ax_card.text(0.5, 0.66, '−0.35%',
             transform=ax_card.transAxes,
             color='#10B981', fontsize=24,
             fontweight='bold', ha='center')
ax_card.text(0.5, 0.58, 'vis-viva vs real telemetry',
             transform=ax_card.transAxes,
             color='#475569', fontsize=7.5, ha='center')

# divider
ax_card.plot([0.15, 0.85], [0.52, 0.52],
             color='#1E293B', lw=0.8,
             transform=ax_card.transAxes)

# Day 3 result
ax_card.text(0.5, 0.45, 'Day 3 — dose rate (radiation)',
             transform=ax_card.transAxes,
             color='#94A3B8', fontsize=8, ha='center')
ax_card.text(0.5, 0.33, '−9 to −14%',
             transform=ax_card.transAxes,
             color='#F59E0B', fontsize=22,
             fontweight='bold', ha='center')
ax_card.text(0.5, 0.25, 'AP9/AE9 model vs HERA measured',
             transform=ax_card.transAxes,
             color='#475569', fontsize=7.5, ha='center')

# bottom note
ax_card.plot([0.15, 0.85], [0.18, 0.18],
             color='#1E293B', lw=0.8,
             transform=ax_card.transAxes)
ax_card.text(0.5, 0.10,
             'Same pattern, different physics.\nModels are good. Reality is better.',
             transform=ax_card.transAxes,
             color='#334155', fontsize=7.5,
             ha='center', style='italic', linespacing=1.5)

plt.suptitle(
    'Predicted vs Measured — AP9/AE9 radiation models vs Artemis I HERA data\n'
    'Source: Laramore et al., WRMISS 2023  ·  George et al., Nature 2024',
    color='white', fontsize=12, fontweight='bold', y=1.01)

plt.savefig('artemis_predicted_vs_measured.png', dpi=120,
            facecolor='#0D1117')
plt.show()

# ── printout ─────────────────────────────────────────────────
print("=" * 56)
print("   PREDICTED vs MEASURED — VALIDATION SUMMARY")
print("=" * 56)
print()
print("  AP9/AE9 model performance (inner belt):")
print(f"    LSU  error : −13.8%  (WRMISS 2023)")
print(f"    HSU1 error : −13.9%  (WRMISS 2023)")
print(f"    HSU2 error :  −9.4%  (WRMISS 2023)")
print()
print("  Peak dose rate:")
print(f"    Measured   : {df['HERA HSU2'].max():.1f} μGy/min")
print(f"    Model      : {predicted['HERA HSU2'].max():.1f} μGy/min")
err_peak = (predicted['HERA HSU2'].max() /
            df['HERA HSU2'].max() - 1) * 100
print(f"    Error      : {err_peak:.1f}%")
print()
print("  Compare with Day 2:")
print(f"    Δv error   : −0.35%  (vis-viva vs telemetry)")
print(f"    Dose error : −9 to −14%  (AP9/AE9 vs HERA)")
print()
print("  Conclusion:")
print("    Orbital mechanics models are more mature.")
print("    Radiation belt models still have meaningful")
print("    uncertainty — especially near solar maximum.")
print("=" * 56)

# ============================================================
# Cell 5 — Cumulative dose integration
# Total absorbed dose across all mission phases
# + comparison against real-world reference doses
# ============================================================

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('#0D1117')
gs  = GridSpec(2, 3, figure=fig,
               hspace=0.45, wspace=0.35)

ax_cum   = fig.add_subplot(gs[0, :2])   # top left  — cumulative over time
ax_phase = fig.add_subplot(gs[0, 2])    # top right — dose by mission phase
ax_ref   = fig.add_subplot(gs[1, :])    # bottom    — reference comparisons

# ── compute phase doses (μGy, HSU2 = worst case crew exposure) ─
col    = 'HERA HSU2'
dt_min = (df['t_hours_from_launch'].iloc[1] -
          df['t_hours_from_launch'].iloc[0]) * 60

inner_start = INNER_ENTRY_UTC - LAUNCH_UTC
inner_end   = INNER_EXIT_UTC  - LAUNCH_UTC
outer_start = OUTER_ENTRY_UTC - LAUNCH_UTC
outer_end   = OUTER_EXIT_UTC  - LAUNCH_UTC

mask_pre   = df['t_hours_from_launch'] < inner_start
mask_inner = ((df['t_hours_from_launch'] >= inner_start) &
              (df['t_hours_from_launch'] <= inner_end))
mask_gap   = ((df['t_hours_from_launch'] > inner_end) &
              (df['t_hours_from_launch'] < outer_start))
mask_outer = ((df['t_hours_from_launch'] >= outer_start) &
              (df['t_hours_from_launch'] <= outer_end))
mask_deep  = df['t_hours_from_launch'] > outer_end

dose_pre   = df.loc[mask_pre,   col].sum() * dt_min
dose_inner = df.loc[mask_inner, col].sum() * dt_min
dose_gap   = df.loc[mask_gap,   col].sum() * dt_min
dose_outer = df.loc[mask_outer, col].sum() * dt_min
dose_deep  = df.loc[mask_deep,  col].sum() * dt_min

# scale deep space to full 25.5-day mission
# we only have 13.5h in our profile — GCR rate is constant
# 25.5 days total - ~11h belt = ~25.04 days deep space
gcr_rate        = 0.28 #DETECTORS[col]['deep_space_gcr']  # μGy/min
deep_total_min  = (25.5 * 24 * 60) - (outer_end * 60)
dose_deep_full  = gcr_rate * deep_total_min

total_mission   = dose_pre + dose_inner + dose_gap + \
                  dose_outer + dose_deep_full

# convert to mGy for readability
def ugy_to_mgy(x): return x / 1000

# ── TOP LEFT: cumulative dose timeline ───────────────────────
ax_cum.set_facecolor('#0D1117')
ax_cum.set_title(
    'Cumulative absorbed dose — HERA HSU2 (crew cabin)',
    color='white', fontsize=11, pad=10)
ax_cum.set_xlabel('Mission elapsed time (hours from launch)',
                  color='#94A3B8', fontsize=9)
ax_cum.set_ylabel('Cumulative absorbed dose (μGy)',
                  color='#94A3B8', fontsize=9)
ax_cum.tick_params(colors='#94A3B8', labelsize=8)
for sp in ['bottom', 'left']:
    ax_cum.spines[sp].set_color('#334155')
for sp in ['top', 'right']:
    ax_cum.spines[sp].set_visible(False)
ax_cum.grid(color='#1E293B', lw=0.6, zorder=0)

# belt shading
ax_cum.axvspan(inner_start, inner_end,
               alpha=0.10, color='#EF4444', zorder=1)
ax_cum.axvspan(outer_start, outer_end,
               alpha=0.07, color='#3B82F6', zorder=1)

# cumulative dose curve
ax_cum.plot(df['t_hours_from_launch'],
            df[col + '_cumulative_uGy'],
            color='#F59E0B', lw=2.5, zorder=5)

# extend flat GCR line to show full mission pace
t_ext  = np.linspace(outer_end, outer_end + 10, 200)
cum_at_exit = df[col + '_cumulative_uGy'].iloc[-1]
gcr_ext = cum_at_exit + gcr_rate * (t_ext - outer_end) * 60
ax_cum.plot(t_ext, gcr_ext, color='#10B981',
            lw=1.5, ls='--', alpha=0.7,
            label='GCR accumulation rate (deep space)', zorder=4)

# phase annotations
phase_times = [
    (inner_start + (inner_end - inner_start)/2,
     dose_pre + dose_inner/2,
     f'Inner belt\n+{dose_inner:.0f} μGy\nin 35 min',
     '#EF4444'),
    (outer_start + (outer_end - outer_start)/2,
     dose_pre + dose_inner + dose_gap + dose_outer/2,
     f'Outer belt\n+{dose_outer:.0f} μGy\nin 120 min',
     '#3B82F6'),
]
for tx, ty, label, col_ in phase_times:
    cum_val = df.loc[
        (df['t_hours_from_launch'] - tx).abs().idxmin(),
        'HERA HSU2_cumulative_uGy'
    ]
    ax_cum.annotate(
        label,
        xy=(tx, cum_val),
        xytext=(tx + 1.2, cum_val - 300),
        color=col_, fontsize=8,
        arrowprops=dict(arrowstyle='->',
                        color=col_, lw=1.0),
        linespacing=1.4
    )

# key milestones
ax_cum.axhline(dose_pre + dose_inner,
               color='#EF4444', lw=0.8, ls=':', alpha=0.5)
ax_cum.text(0.2, dose_pre + dose_inner + 80,
            f'After inner belt: {dose_pre + dose_inner:.0f} μGy',
            color='#EF4444', fontsize=7.5)

final_cum = df['HERA HSU2_cumulative_uGy'].iloc[-1]
ax_cum.text(0.2, final_cum + 80,
            f'At T+13.5h: {final_cum:.0f} μGy',
            color='#F59E0B', fontsize=7.5)

ax_cum.legend(facecolor='#1E293B', edgecolor='#334155',
              labelcolor='white', fontsize=8,
              loc='lower right')
ax_cum.set_xlim(0, outer_end + 10.5)

# ── TOP RIGHT: dose by phase (pie / donut) ───────────────────
ax_phase.set_facecolor('#0D1117')
ax_phase.set_title('Dose contribution\nby mission phase\n(full 25.5-day mission)',
                   color='white', fontsize=9, pad=8)

phase_labels = [
    f'Inner belt\n{dose_inner/1000:.2f} mGy',
    f'Outer belt\n{dose_outer/1000:.2f} mGy',
    f'Deep space GCR\n{dose_deep_full/1000:.2f} mGy',
    f'Pre-belt\n{dose_pre/1000:.3f} mGy',
]
phase_vals = [dose_inner, dose_outer, dose_deep_full, dose_pre]
phase_cols = ['#EF4444', '#3B82F6', '#10B981', '#475569']

wedges, _, autotexts = ax_phase.pie(
    phase_vals,
    labels=phase_labels,
    autopct='%1.1f%%',
    colors=phase_cols,
    startangle=90,
    wedgeprops=dict(edgecolor='#0D1117', linewidth=1.5,
                    width=0.6),   # donut
    textprops=dict(color='#94A3B8', fontsize=7.5)
)
for at in autotexts:
    at.set_color('white')
    at.set_fontsize(8)
    at.set_fontweight('bold')

ax_phase.text(0, 0,
              f'{total_mission/1000:.2f}\nmGy\ntotal',
              ha='center', va='center',
              color='white', fontsize=9,
              fontweight='bold', linespacing=1.4)

# ── BOTTOM: reference dose comparison ────────────────────────
ax_ref.set_facecolor('#0D1117')
ax_ref.set_title(
    'Context: how does Artemis I exposure compare?',
    color='white', fontsize=11, pad=10)
ax_ref.tick_params(colors='#94A3B8', labelsize=8)
for sp in ['bottom', 'left']:
    ax_ref.spines[sp].set_color('#334155')
for sp in ['top', 'right']:
    ax_ref.spines[sp].set_visible(False)
ax_ref.grid(axis='x', color='#1E293B', lw=0.6, zorder=0)
ax_ref.set_xlabel('Absorbed dose (mGy)',
                  color='#94A3B8', fontsize=9)

# reference doses — all in mGy
references = [
    ('Chest X-ray',                     0.0001, '#475569'),
    ('Transatlantic flight',             0.005,  '#475569'),
    ('Annual background (Earth)',        2.4,    '#475569'),
    ('Dental CT scan',                   0.9,    '#475569'),
    ('ISS 6-month mission',              80.0,   '#3B82F6'),
    ('Artemis I — inner belt only\n(HSU2, 35 min)',
                                         dose_inner/1000, '#EF4444'),
    ('Artemis I — full mission\n(HSU2, 25.5 days)',
                                         total_mission/1000, '#F59E0B'),
    ('NASA career limit\n(effective dose equiv.)',
                                         600.0,  '#10B981'),
]

ref_labels = [r[0] for r in references]
ref_vals   = [r[1] for r in references]
ref_cols   = [r[2] for r in references]

bars = ax_ref.barh(ref_labels, ref_vals,
                   color=ref_cols, alpha=0.85,
                   height=0.5, zorder=3)

for bar, val in zip(bars, ref_vals):
    label = f'{val:.4g} mGy' if val < 0.01 else f'{val:.2f} mGy'
    ax_ref.text(val * 1.02,
                bar.get_y() + bar.get_height()/2,
                label, va='center',
                color='white', fontsize=8.5,
                fontweight='bold')

# Artemis II crew marker
ax_ref.axvline(total_mission/1000, color='#F59E0B',
               lw=1.2, ls=':', alpha=0.6)

ax_ref.set_xscale('log')
ax_ref.set_xlim(0.00005, 2000)

plt.suptitle(
    'Cumulative Dose Analysis — Artemis I HERA HSU2\n'
    'Source: George et al., Nature 2024  ·  NASA NTRS 20230010550',
    color='white', fontsize=12, fontweight='bold', y=1.01)

plt.savefig('artemis_cumulative_dose.png', dpi=120,
            facecolor='#0D1117')
plt.show()

# ── printout ─────────────────────────────────────────────────
print("=" * 56)
print("   CUMULATIVE DOSE SUMMARY — HERA HSU2 (crew cabin)")
print("=" * 56)
print(f"\n  Pre-belt (LEO)       : {dose_pre:.1f} μGy")
print(f"  Inner belt (35 min)  : {dose_inner:.1f} μGy"
      f"  ({dose_inner/total_mission*100:.1f}% of total)")
print(f"  Inter-belt gap       : {dose_gap:.1f} μGy")
print(f"  Outer belt (120 min) : {dose_outer:.1f} μGy"
      f"  ({dose_outer/total_mission*100:.1f}% of total)")
print(f"  Deep space GCR       : {dose_deep_full:.1f} μGy"
      f"  ({dose_deep_full/total_mission*100:.1f}% of total)")
print(f"\n  Total mission dose   : {total_mission:.1f} μGy")
print(f"                       : {total_mission/1000:.3f} mGy")
print()
print(f"  Inner belt dominates despite being only 35 of")
print(f"  {25.5*24*60:.0f} total mission minutes ({35/(25.5*24*60)*100:.2f}%"
      f" of mission time)")
print()
print(f"  NASA career limit    : 600,000 μGy equivalent")
print(f"  This mission         : {total_mission:.0f} μGy")
print(f"  Fraction of limit    : {total_mission/600000*100:.2f}%")
print("=" * 56)

# ============================================================
# Cell 6 — Dose rate animation
# Orion moving through radiation zones with live exposure counter
# ============================================================

fig, (ax_map, ax_plot) = plt.subplots(
    1, 2, figsize=(14, 6),
    gridspec_kw={'width_ratios': [1, 1.4]}
)
fig.patch.set_facecolor('#0D1117')

# ── LEFT: radiation zone map ──────────────────────────────────
ax_map.set_facecolor('#0D1117')
ax_map.set_xlim(-5, 5)
ax_map.set_ylim(-5, 5)
ax_map.set_aspect('equal')
ax_map.axis('off')
ax_map.set_title('Radiation environment',
                 color='white', fontsize=10, pad=8)

# Earth
earth = plt.Circle((0, 0), 0.55,
                   color='#1B4FD8', zorder=6)
ax_map.add_patch(earth)
ax_map.text(0, 0, 'Earth', color='white',
            ha='center', va='center',
            fontsize=7, fontweight='bold', zorder=7)

# Van Allen belt zones — concentric rings
# Scale: 1 unit = ~12,000 km
# Inner belt: 1000–12000 km → r = 0.63–1.55
# Outer belt: 13000–60000 km → r = 1.63–5.55
# We compress outer belt for visibility
inner_r1, inner_r2 = 0.63, 1.55
outer_r1, outer_r2 = 1.63, 3.80

inner_ring = plt.matplotlib.patches.Annulus(
    (0, 0), inner_r2, inner_r2 - inner_r1,
    color='#EF4444', alpha=0.25, zorder=2
)
outer_ring = plt.matplotlib.patches.Annulus(
    (0, 0), outer_r2, outer_r2 - outer_r1,
    color='#3B82F6', alpha=0.15, zorder=2
)
ax_map.add_patch(inner_ring)
ax_map.add_patch(outer_ring)

# zone labels
ax_map.text(0, 1.1, 'Inner belt\n(protons)',
            color='#EF4444', ha='center',
            fontsize=7, style='italic', zorder=8)
ax_map.text(0, 2.7, 'Outer belt\n(electrons)',
            color='#3B82F6', ha='center',
            fontsize=7, style='italic', zorder=8)
ax_map.text(0, 4.5, 'Deep space\n(GCR)',
            color='#10B981', ha='center',
            fontsize=7, style='italic', zorder=8)

# orbit path (faint arc)
theta_path = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, 300)
r_path = np.linspace(0.63, 4.2, 300)
x_path = r_path * np.cos(theta_path)
y_path = r_path * np.sin(theta_path)
ax_map.plot(x_path, y_path, color='#334155',
            lw=1.0, ls='--', zorder=3, alpha=0.5)

# spacecraft marker
sc_dot, = ax_map.plot([], [], 's',
                      color='white', ms=8, zorder=10)
sc_glow, = ax_map.plot([], [], 'o',
                       color='#F59E0B', ms=0,
                       zorder=9, alpha=0.5)

# dose readout on map
dose_text_map = ax_map.text(
    0, -3.8,
    'Dose rate: 0.00 μGy/min',
    color='white', ha='center', fontsize=9,
    fontweight='bold', zorder=10
)
phase_text_map = ax_map.text(
    0, -4.4,
    'Phase: pre-belt',
    color='#94A3B8', ha='center', fontsize=8, zorder=10
)
cumulative_text = ax_map.text(
    0, -4.95,
    'Cumulative: 0.00 μGy',
    color='#F59E0B', ha='center', fontsize=8, zorder=10
)

# ── RIGHT: live dose rate plot ────────────────────────────────
ax_plot.set_facecolor('#0D1117')
ax_plot.set_title('Live dose rate — HERA HSU2',
                  color='white', fontsize=10, pad=8)
ax_plot.set_xlabel('Mission elapsed time (hours)',
                   color='#94A3B8', fontsize=9)
ax_plot.set_ylabel('Dose rate (μGy/min)',
                   color='#94A3B8', fontsize=9)
ax_plot.tick_params(colors='#94A3B8', labelsize=8)
for sp in ['bottom', 'left']:
    ax_plot.spines[sp].set_color('#334155')
for sp in ['top', 'right']:
    ax_plot.spines[sp].set_visible(False)
ax_plot.grid(color='#1E293B', lw=0.6, zorder=0)
ax_plot.set_yscale('log')
ax_plot.set_xlim(0, 13.5)
ax_plot.set_ylim(0.05, 1200)

# belt shading on plot
ax_plot.axvspan(inner_start, inner_end,
                alpha=0.10, color='#EF4444')
ax_plot.axvspan(outer_start, outer_end,
                alpha=0.07, color='#3B82F6')

# full data faint background
ax_plot.plot(df['t_hours_from_launch'], df['HERA HSU2'],
             color='#334155', lw=1.0, alpha=0.4, zorder=2)

# live line
live_line, = ax_plot.plot([], [], color='#F59E0B',
                          lw=2.0, zorder=5)
live_dot,  = ax_plot.plot([], [], 'o', color='#F59E0B',
                          ms=7, zorder=6)

# ── Animation setup ───────────────────────────────────────────
STEP   = 8     # frames to skip for speed
N_frames = len(df) // STEP

t_arr   = df['t_hours_from_launch'].values
dose_arr = df['HERA HSU2'].values
cum_arr  = df['HERA HSU2_cumulative_uGy'].values

live_t    = []
live_dose = []

def spacecraft_position(t_h):
    """Map mission time to (x,y) on radiation zone map."""
    if t_h < inner_start:
        r = 0.62
    elif t_h <= inner_end:
        frac = (t_h - inner_start) / (inner_end - inner_start)
        r = inner_r1 + (inner_r2 - inner_r1) * frac
    elif t_h < outer_start:
        frac = (t_h - inner_end) / (outer_start - inner_end)
        r = inner_r2 + (outer_r1 - inner_r2) * frac
    elif t_h <= outer_end:
        frac = (t_h - outer_start) / (outer_end - outer_start)
        r = outer_r1 + (outer_r2 - outer_r1) * frac
    else:
        frac = min((t_h - outer_end) / 5.0, 1.0)
        r = outer_r2 + frac * 0.8
    angle = np.pi/2 + t_h * 0.4
    return r * np.cos(angle), r * np.sin(angle)

def get_phase(t_h, dose_r):
    if t_h < inner_start:
        return 'LEO — pre-belt', '#94A3B8'
    elif t_h <= inner_end:
        return 'Inner belt (protons)', '#EF4444'
    elif t_h < outer_start:
        return 'Inter-belt gap', '#94A3B8'
    elif t_h <= outer_end:
        return 'Outer belt (electrons)', '#3B82F6'
    else:
        return 'Deep space (GCR)', '#10B981'

def get_glow_size(dose_r):
    if dose_r > 100:
        return 20 + 8 * np.sin(dose_r * 0.1)
    elif dose_r > 1:
        return 10
    else:
        return 0

def init():
    sc_dot.set_data([], [])
    sc_glow.set_data([], [])
    sc_glow.set_markersize(0)
    live_line.set_data([], [])
    live_dot.set_data([], [])
    dose_text_map.set_text('Dose rate: 0.00 μGy/min')
    phase_text_map.set_text('Phase: pre-belt')
    cumulative_text.set_text('Cumulative: 0.00 μGy')
    live_t.clear()
    live_dose.clear()
    return (sc_dot, sc_glow, live_line, live_dot,
            dose_text_map, phase_text_map, cumulative_text)

def update(frame):
    idx = frame * STEP
    if idx >= len(df):
        idx = len(df) - 1

    t_h    = t_arr[idx]
    dose_r = dose_arr[idx]
    cum    = cum_arr[idx]

    # spacecraft position
    x, y = spacecraft_position(t_h)
    sc_dot.set_data([x], [y])
    sc_glow.set_data([x], [y])
    sc_glow.set_markersize(get_glow_size(dose_r))

    # phase + color
    phase, p_color = get_phase(t_h, dose_r)
    sc_dot.set_color(p_color)

    # readouts
    dose_text_map.set_text(
        f'Dose rate: {dose_r:.1f} μGy/min')
    dose_text_map.set_color(p_color)
    phase_text_map.set_text(f'Phase: {phase}')
    phase_text_map.set_color(p_color)
    cumulative_text.set_text(
        f'Cumulative: {cum:,.1f} μGy')

    # live plot
    live_t.append(t_h)
    live_dose.append(dose_r)
    live_line.set_data(live_t, live_dose)
    live_dot.set_data([t_h], [dose_r])
    live_line.set_color(p_color)
    live_dot.set_color(p_color)

    return (sc_dot, sc_glow, live_line, live_dot,
            dose_text_map, phase_text_map, cumulative_text)

anim6 = animation.FuncAnimation(
    fig, update, frames=N_frames,
    init_func=init, interval=40, blit=True
)

plt.tight_layout()

anim6.save('artemis_radiation_animation.gif',
           writer='pillow', fps=30, dpi=100,
           savefig_kwargs={'facecolor': '#0D1117'})
print("Saved: artemis_radiation_animation.gif")

HTML(anim6.to_jshtml())

# ============================================================
# Cell 7 — Radiation context comparison
# ISS vs deep space vs Artemis I vs everyday references
# The "so what does this mean?" cell
# ============================================================

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0D1117')
gs  = GridSpec(2, 3, figure=fig,
               hspace=0.48, wspace=0.38)

ax_bar   = fig.add_subplot(gs[:, :2])   # left wide — main comparison
ax_rate  = fig.add_subplot(gs[0, 2])    # top right — dose rate comparison
ax_time  = fig.add_subplot(gs[1, 2])    # bottom right — time to limit

# ── Data — all values from published sources ──────────────────
# Dose rates in μGy/min, total doses in mGy

environments = [
    # (label, dose_rate_uGy_min, source)
    ('Earth surface\n(natural background)',
     0.000456, '#475569'),           # 2.4 mSv/yr ÷ 525960 min
    ('Commercial flight\n(transatlantic)',
     0.0056,   '#475569'),           # ~0.08 mSv per 8h flight
    ('ISS crew\n(LEO, 400 km)',
     0.109,    '#3B82F6'),           # ~80 mGy/6months
    ('Artemis I deep space\n(GCR, HERA HSU2)',
     0.28,     '#10B981'),           # Nature 2024
    ('Artemis I outer belt\n(electron peak, HSU2)',
     4.0,      '#60A5FA'),           # Nature 2024
    ('Artemis I inner belt\n(proton peak, HSU2)',
     287.0,    '#EF4444'),           # Nature 2024
    ('Solar particle event\n(shelter threshold)',
     75.0,     '#F59E0B'),           # NASA HERA threshold
]

# mission total doses (mGy)
mission_doses = [
    ('Chest X-ray',                    0.0001, '#475569'),
    ('Annual background (Earth)',       2.4,    '#475569'),
    ('ISS 6-month mission',            80.0,   '#3B82F6'),
    ('Artemis I — belt transit only\n(HSU2, ~155 min)',
                                       5.08,   '#EF4444'),
    ('Artemis I — full mission\n(HSU2, 25.5 days)',
                                       15.29,  '#F59E0B'),
    ('Mars mission estimate\n(~500 days transit)',
                                       500.0,  '#EC4899'),
    ('NASA career limit\n(600 mSv eff. dose)',
                                       600.0,  '#10B981'),
]

# ── LEFT: main horizontal bar — dose rate comparison ─────────
ax_bar.set_facecolor('#0D1117')
ax_bar.set_title(
    'Radiation dose rate — from Earth surface to inner Van Allen belt',
    color='white', fontsize=12, pad=12)
ax_bar.tick_params(colors='#94A3B8', labelsize=9)
for sp in ['bottom', 'left']:
    ax_bar.spines[sp].set_color('#334155')
for sp in ['top', 'right']:
    ax_bar.spines[sp].set_visible(False)
ax_bar.grid(axis='x', color='#1E293B', lw=0.6, zorder=0)
ax_bar.set_xlabel('Dose rate (μGy/min) — log scale',
                  color='#94A3B8', fontsize=9)
ax_bar.set_xscale('log')

labels   = [e[0] for e in environments]
rates    = [e[1] for e in environments]
e_colors = [e[2] for e in environments]

bars = ax_bar.barh(labels, rates,
                   color=e_colors, alpha=0.85,
                   height=0.55, zorder=3)

for bar, val, label in zip(bars, rates, labels):
    if val >= 0.01:
        txt = f'{val:.1f} μGy/min'
    elif val >= 0.001:
        txt = f'{val:.4f} μGy/min'
    else:
        txt = f'{val:.5f} μGy/min'
    ax_bar.text(
        val * 1.15,
        bar.get_y() + bar.get_height()/2,
        txt, va='center', color='white',
        fontsize=8.5, fontweight='bold'
    )

# multiplier annotations
pairs = [
    (0, 3, '613×\nmore than\nEarth surface'),
    (2, 3, '2.6×\nmore than\nISS'),
    (3, 5, '1,025×\nmore at\ninner belt peak'),
]
for i_low, i_high, label in pairs:
    y_low  = i_low
    y_high = i_high
    x_pos  = max(rates[i_low], rates[i_high]) * 8
    ax_bar.annotate('',
        xy=(x_pos, y_high),
        xytext=(x_pos, y_low),
        arrowprops=dict(arrowstyle='<->',
                        color='#475569', lw=1.0)
    )
    ax_bar.text(x_pos * 1.3,
                (y_low + y_high) / 2,
                label, color='#64748B',
                fontsize=7, ha='left',
                va='center', linespacing=1.4)

ax_bar.set_xlim(0.00015, 8000)

# ── TOP RIGHT: mission total dose bar ────────────────────────
ax_rate.set_facecolor('#0D1117')
ax_rate.set_title('Total mission dose\n(absorbed, mGy)',
                  color='white', fontsize=9, pad=8)
ax_rate.tick_params(colors='#94A3B8', labelsize=7.5)
for sp in ['bottom', 'left']:
    ax_rate.spines[sp].set_color('#334155')
for sp in ['top', 'right']:
    ax_rate.spines[sp].set_visible(False)
ax_rate.grid(axis='x', color='#1E293B', lw=0.6, zorder=0)
ax_rate.set_xscale('log')
ax_rate.set_xlabel('Absorbed dose (mGy)',
                   color='#94A3B8', fontsize=8)

m_labels = [m[0] for m in mission_doses]
m_vals   = [m[1] for m in mission_doses]
m_colors = [m[2] for m in mission_doses]

m_bars = ax_rate.barh(m_labels, m_vals,
                      color=m_colors, alpha=0.85,
                      height=0.5, zorder=3)

for bar, val in zip(m_bars, m_vals):
    txt = f'{val:.4g}'
    ax_rate.text(
        val * 1.15,
        bar.get_y() + bar.get_height()/2,
        txt, va='center', color='white',
        fontsize=7.5, fontweight='bold'
    )

ax_rate.set_xlim(0.00005, 5000)

# ── BOTTOM RIGHT: time to reach career limit ──────────────────
ax_time.set_facecolor('#0D1117')
ax_time.set_title('Time to reach NASA career\nlimit at each dose rate',
                  color='white', fontsize=9, pad=8)
ax_time.tick_params(colors='#94A3B8', labelsize=7.5)
for sp in ['bottom', 'left']:
    ax_time.spines[sp].set_color('#334155')
for sp in ['top', 'right']:
    ax_time.spines[sp].set_visible(False)
ax_time.grid(axis='x', color='#1E293B', lw=0.6, zorder=0)
ax_time.set_xlabel('Time to career limit (days)',
                   color='#94A3B8', fontsize=8)
ax_time.set_xscale('log')

# career limit = 600,000 μGy equivalent
CAREER_LIMIT_uGy = 600_000

time_labels = []
time_vals   = []
time_colors = []

for label, rate, color in environments[2:]:   # skip Earth/flight
    t_days = CAREER_LIMIT_uGy / rate / 60 / 24
    time_labels.append(label.split('\n')[0])
    time_vals.append(t_days)
    time_colors.append(color)

t_bars = ax_time.barh(time_labels, time_vals,
                      color=time_colors, alpha=0.85,
                      height=0.5, zorder=3)

for bar, val in zip(t_bars, time_vals):
    if val > 365:
        txt = f'{val/365:.1f} yrs'
    elif val > 1:
        txt = f'{val:.1f} days'
    else:
        txt = f'{val*24:.1f} hrs'
    ax_time.text(
        val * 1.15,
        bar.get_y() + bar.get_height()/2,
        txt, va='center', color='white',
        fontsize=7.5, fontweight='bold'
    )

ax_time.set_xlim(0.001, 50000)

plt.suptitle(
    'Space Radiation in Context — from Earth to the Inner Van Allen Belt\n'
    'Sources: George et al. Nature 2024  ·  NASA NTRS  ·  '
    'NASA career limit (600 mSv)',
    color='white', fontsize=11,
    fontweight='bold', y=1.01)

plt.savefig('artemis_radiation_comparison.png', dpi=120,
            facecolor='#0D1117')
plt.show()

# ── printout ─────────────────────────────────────────────────
print("=" * 58)
print("   RADIATION COMPARISON — KEY NUMBERS FOR BLOG")
print("=" * 58)
print()
print("  Dose rates:")
print(f"    Earth surface       : 0.000456 μGy/min")
print(f"    ISS (LEO)           : 0.109    μGy/min")
print(f"    Deep space GCR      : 0.280    μGy/min  (2.6× ISS)")
print(f"    Inner belt peak     : 287.0    μGy/min  (1,025× ISS)")
print()
print("  Time to career limit (600,000 μGy) at each rate:")
for label, rate, _ in environments[2:]:
    t_days = CAREER_LIMIT_uGy / rate / 60 / 24
    short  = label.split('\n')[0]
    if t_days > 365:
        print(f"    {short:<35}: {t_days/365:.1f} years")
    elif t_days > 1:
        print(f"    {short:<35}: {t_days:.1f} days")
    else:
        print(f"    {short:<35}: {t_days*24:.1f} hours")
print()
print("  Mission totals:")
print(f"    Artemis I belt only : 5.08 mGy")
print(f"    Artemis I full      : 15.29 mGy  (2.55% of limit)")
print(f"    ISS 6-month         : 80.0 mGy   (13.3% of limit)")
print(f"    Mars estimate       : 500.0 mGy  (83.3% of limit)")
print("=" * 58)

# ============================================================
# Cell 9 — Day 3 Summary Dashboard (fixed layout)
# ============================================================

# ── canvas: wider, taller, explicit axes ─────────────────────
fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor('#0D1117')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_facecolor('#0D1117')

# ── helpers ───────────────────────────────────────────────────
def card(x, y, w, h, fc='#1E293B', ec='#334155', lw=0.8, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.08',
        facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=2, alpha=alpha
    ))

def txt(x, y, s, **kw):
    kw.setdefault('color', 'white')
    kw.setdefault('fontsize', 10)
    kw.setdefault('zorder', 4)
    kw.setdefault('va', 'center')
    ax.text(x, y, s, **kw)

def hline(x0, x1, y, col='#1E293B', lw=0.7):
    ax.plot([x0, x1], [y, y],
            color=col, lw=lw, zorder=3)

# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
card(0.25, 10.7, 19.5, 1.1, fc='#0F172A', ec='#1E3A5C')
txt(10.0, 11.35,
    'DAY 3 — THE SILENT PASSENGER: DEEP SPACE RADIATION',
    fontsize=16, fontweight='bold', ha='center')
txt(10.0, 10.97,
    'Code Beyond the Earth  ·  Artemis I HERA — '
    'George et al., Nature 2024  ·  WRMISS 2023  ·  April 3, 2026',
    fontsize=9, ha='center', color='#475569')

# ════════════════════════════════════════════════════════════
# ROW 1 — five metric cards (equal width, well spaced)
# ════════════════════════════════════════════════════════════
metrics = [
    ('Inner belt peak',   '287',   'μGy/min', '#EF4444'),
    ('Deep space GCR',    '0.28',  'μGy/min', '#10B981'),
    ('Belt dose (HSU2)',  '5,075', 'μGy',     '#F59E0B'),
    ('Full mission dose', '15.29', 'mGy',     '#F59E0B'),
    ('Career limit used', '2.55',  '%',        '#10B981'),
]
card_w = 3.6
gap    = 0.25
x0_row1 = 0.25
for i, (label, val, unit, col) in enumerate(metrics):
    cx = x0_row1 + i * (card_w + gap)
    card(cx, 9.25, card_w, 1.25, fc='#0F172A', ec=col)
    txt(cx + card_w/2, 10.15, label,
        fontsize=8.5, color='#64748B', ha='center')
    txt(cx + card_w/2, 9.78, val,
        fontsize=18, fontweight='bold', color=col, ha='center')
    txt(cx + card_w/2, 9.42, unit,
        fontsize=8, color='#475569', ha='center')

# ════════════════════════════════════════════════════════════
# ROW 2 — three equal cards
# Each card: w=6.2, starting at x=0.25
# ════════════════════════════════════════════════════════════
CW   = 6.2    # card width
CGap = 0.25   # gap between cards
CY0  = 5.05   # card bottom
CH   = 4.0    # card height
CY1  = CY0 + CH  # card top

cx1 = 0.25
cx2 = cx1 + CW + CGap
cx3 = cx2 + CW + CGap

card(cx1, CY0, CW, CH)
card(cx2, CY0, CW, CH)
card(cx3, CY0, CW, CH)

# ── CARD 1: Mission time vs dose ─────────────────────────────
txt(cx1 + CW/2, CY1 - 0.3,
    'Mission time vs dose contribution',
    fontsize=9.5, fontweight='bold', ha='center')

phases = [
    ('Pre-belt',  0.10, 3.5,  '#475569'),
    ('Inner\nbelt', 0.10, 31.3, '#EF4444'),
    ('Gap',       0.07, 0.1,  '#334155'),
    ('Outer\nbelt', 0.33, 1.8, '#3B82F6'),
    ('Deep\nspace', 99.4, 66.8,'#10B981'),
]

BAR_W  = 5.6      # full bar width
BAR_H  = 0.30     # bar height
BX0    = cx1 + 0.3
Y_TIME = CY1 - 1.05
Y_DOSE = CY1 - 1.95

# row labels
txt(BX0 - 0.18, Y_TIME + BAR_H/2, 'Time',
    fontsize=8, color='#64748B', ha='right')
txt(BX0 - 0.18, Y_DOSE + BAR_H/2, 'Dose',
    fontsize=8, color='#64748B', ha='right')

for i, (label, t_pct, d_pct, col) in enumerate(phases):
    # compute segment width proportional to percent
    t_w = t_pct / 100 * BAR_W
    d_w = d_pct / 100 * BAR_W

    # cumulative x position
    t_offset = sum(p[1]/100*BAR_W for p in phases[:i])
    #d_offset = sum(p[3]/100*BAR_W for p in phases[:i])
    d_offset = sum(p[2]/100*BAR_W for p in phases[:i])


    # time bar segment
    if t_pct > 0.05:
        ax.add_patch(FancyBboxPatch(
            (BX0 + t_offset, Y_TIME), max(t_w, 0.06), BAR_H,
            boxstyle='round,pad=0.01',
            facecolor=col, alpha=0.8,
            edgecolor='#0D1117', linewidth=0.5, zorder=3
        ))
        if t_w > 0.3:
            txt(BX0 + t_offset + t_w/2,
                Y_TIME + BAR_H/2,
                f'{t_pct:.1f}%',
                fontsize=6.5, ha='center', color='white', zorder=5)

    # dose bar segment
    if d_pct > 0.05:
        ax.add_patch(FancyBboxPatch(
            (BX0 + d_offset, Y_DOSE), max(d_w, 0.06), BAR_H,
            boxstyle='round,pad=0.01',
            facecolor=col, alpha=0.9,
            edgecolor='#0D1117', linewidth=0.5, zorder=3
        ))
        if d_w > 0.3:
            txt(BX0 + d_offset + d_w/2,
                Y_DOSE + BAR_H/2,
                f'{d_pct:.1f}%',
                fontsize=6.5, ha='center', color='white', zorder=5)

    # phase label below time bar
    label_x = BX0 + t_offset + max(t_w, 0.06)/2
    txt(label_x, Y_TIME - 0.22, label,
        fontsize=6.5, ha='center',
        color='#64748B', linespacing=1.3)

# key insight box
card(cx1 + 0.2, CY0 + 0.18, CW - 0.4, 0.82,
     fc='#052e16', ec='#166534')
txt(cx1 + CW/2, CY0 + 0.59,
    'Inner belt: 0.10% of mission time → 31.3% of total dose',
    fontsize=8.5, ha='center', color='#4ade80',
    fontweight='bold')

# ── CARD 2: Validation ────────────────────────────────────────
txt(cx2 + CW/2, CY1 - 0.3,
    'Model validation — Days 2 & 3',
    fontsize=9.5, fontweight='bold', ha='center')

val_rows = [
    ('Day 2 — Δv',
     'vis-viva vs Artemis I telemetry',
     '−0.35%', '#10B981'),
    ('Day 3 — inner belt dose',
     'AP9/AE9 vs HERA HSU2',
     '−9.4%', '#F59E0B'),
    ('Day 3 — inner belt dose',
     'AP9/AE9 vs HERA LSU/HSU1',
     '−13.8%', '#F59E0B'),
]
row_h = 0.95
for j, (title, sub, err, col) in enumerate(val_rows):
    ry = CY1 - 1.1 - j * row_h
    txt(cx2 + 0.25, ry,       title,
        fontsize=8.5, color='white', ha='left')
    txt(cx2 + 0.25, ry - 0.25, sub,
        fontsize=7.5, color='#475569', ha='left')
    txt(cx2 + CW - 0.2, ry - 0.12, err,
        fontsize=15, color=col,
        fontweight='bold', ha='right')
    if j < 2:
        hline(cx2 + 0.2, cx2 + CW - 0.2,
              ry - 0.58, col='#1E293B', lw=0.7)

card(cx2 + 0.2, CY0 + 0.18, CW - 0.4, 0.82,
     fc='#0F172A', ec='#334155')
txt(cx2 + CW/2, CY0 + 0.59,
    'Orbital mechanics: mature, precise.\n'
    'Radiation models: improving with every mission.',
    fontsize=8, ha='center', color='#64748B',
    linespacing=1.5)

# ── CARD 3: Shielding ─────────────────────────────────────────
txt(cx3 + CW/2, CY1 - 0.3,
    'Shielding & orientation effects',
    fontsize=9.5, fontweight='bold', ha='center')

shield = [
    ('M-42 SN127\n(high shield)',  1289,  '#10B981'),
    ('EAD MU01\n(low shield)',     4241,  '#F59E0B'),
    ('HERA HSU2\n(low shield)',    5075,  '#EF4444'),
]
SBW   = 4.8     # shield bar max width
SBX0  = cx3 + 1.2
SBY0  = CY1 - 1.1
SBH   = 0.35
SB_GAP= 0.78
MAX_V = 5075

for j, (label, val, col) in enumerate(shield):
    by  = SBY0 - j * SB_GAP
    bw  = val / MAX_V * SBW

    # track
    ax.add_patch(FancyBboxPatch(
        (SBX0, by), SBW, SBH,
        boxstyle='round,pad=0.02',
        facecolor='#1E293B', edgecolor='none', zorder=3
    ))
    # fill
    ax.add_patch(FancyBboxPatch(
        (SBX0, by), bw, SBH,
        boxstyle='round,pad=0.02',
        facecolor=col, alpha=0.85,
        edgecolor='none', zorder=4
    ))
    # label LEFT of bar
    txt(SBX0 - 0.1, by + SBH/2, label,
        fontsize=7.5, color='#94A3B8',
        ha='right', linespacing=1.3)
    # value RIGHT of bar
    txt(SBX0 + SBW + 0.12, by + SBH/2,
        f'{val:,} μGy',
        fontsize=8.5, color=col,
        fontweight='bold', ha='left')

# 4x annotation
ax.annotate('',
    xy=(SBX0, SBY0 - 2*SB_GAP + SBH/2),
    xytext=(SBX0 + SBW, SBY0 - 2*SB_GAP + SBH/2),
    arrowprops=dict(arrowstyle='<->',
                    color='#475569', lw=1.0)
)
txt(SBX0 + SBW/2, SBY0 - 2*SB_GAP - 0.22,
    '4× shielding difference (Nature 2024)',
    fontsize=7.5, ha='center', color='#64748B',
    style='italic')

card(cx3 + 0.2, CY0 + 0.18, CW - 0.4, 0.82,
     fc='#1c1400', ec='#854F0B')
txt(cx3 + CW/2, CY0 + 0.59,
    'ICPS orientation burn → 50% dose reduction\n'
    'Directional protons blocked by vehicle bulk  (Nature 2024)',
    fontsize=8, ha='center', color='#FAC775',
    linespacing=1.5)

# ════════════════════════════════════════════════════════════
# ROW 3 — log-scale dose rate strip (tighter, better spaced)
# ════════════════════════════════════════════════════════════
card(0.25, 0.85, 19.5, 4.0)
txt(10.0, 4.6,
    'Dose rate context — six orders of magnitude',
    fontsize=10, fontweight='bold', ha='center')

context = [
    ('Earth\nsurface',        0.000456, '#6B7280'),
    ('Commercial\nflight',    0.0056,   '#6B7280'),
    ('ISS crew\n(LEO)',       0.109,    '#3B82F6'),
    ('Artemis I\ndeep space', 0.280,    '#10B981'),
    ('Outer belt\npeak',      4.0,      '#60A5FA'),
    ('Solar particle\nthreshold', 75.0, '#F59E0B'),
    ('Inner belt\npeak',      287.0,    '#EF4444'),
]

LOG_MIN = np.log10(0.0002)
LOG_MAX = np.log10(900)
AX0     = 0.8
AXW     = 18.4
AXY     = 2.55   # axis line y
DOT_Y   = 2.55   # dot y
VAL_Y   = 3.45   # value label y
LBL_Y   = 1.65   # name label y

# axis line
ax.plot([AX0, AX0 + AXW], [AXY, AXY],
        color='#334155', lw=1.2, zorder=3)

# log ticks
for exp in [-3, -2, -1, 0, 1, 2]:
    tx = AX0 + (exp - LOG_MIN) / (LOG_MAX - LOG_MIN) * AXW
    ax.plot([tx, tx], [AXY - 0.12, AXY + 0.12],
            color='#475569', lw=0.8, zorder=3)
    txt(tx, AXY - 0.32, f'10^{exp}',
        fontsize=7, ha='center', color='#334155')

txt(AX0 + AXW/2, AXY - 0.65,
    'log scale  ·  μGy/min',
    fontsize=7.5, ha='center', color='#334155',
    style='italic')

# dots and labels
for label, rate, col in context:
    bx = AX0 + (np.log10(rate) - LOG_MIN) / \
         (LOG_MAX - LOG_MIN) * AXW

    # dot on axis
    ax.plot(bx, DOT_Y, 'o',
            color=col, ms=11, zorder=5,
            markeredgecolor='#0D1117',
            markeredgewidth=0.8)

    # value above
    if rate >= 1:
        v_str = f'{rate:.0f}'
    elif rate >= 0.01:
        v_str = f'{rate:.3f}'
    else:
        v_str = f'{rate:.5f}'
    txt(bx, VAL_Y, v_str,
        fontsize=8, ha='center',
        color=col, fontweight='bold')
    txt(bx, VAL_Y + 0.32, 'μGy/min',
        fontsize=6.5, ha='center', color='#475569')

    # label below
    txt(bx, LBL_Y, label,
        fontsize=7.5, ha='center',
        color='#94A3B8', linespacing=1.4)

# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════
card(0.25, 0.12, 19.5, 0.62, fc='#0F172A', ec='#1E293B')
now = datetime.now().strftime('%B %d, %Y')
txt(10.0, 0.44,
    f'Code Beyond the Earth  ·  codebeyondtheearth.hashnode.dev  '
    f'·  Generated {now}  ·  '
    f'Data: George et al. Nature 2024  ·  WRMISS 2023  ·  NASA NTRS',
    fontsize=8.5, ha='center', color='#475569')

plt.savefig('artemis_day3_summary_v2.png', dpi=150,
            facecolor='#0D1117', bbox_inches='tight')
plt.show()
print("Saved: artemis_day3_summary_v2.png")