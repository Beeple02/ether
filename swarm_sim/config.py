NUM_DRONES = 80
MAX_SPEED  = 3.0
MAX_FORCE  = 0.3
PERCEPTION_RADIUS  = 80
SEPARATION_RADIUS  = 30
WORLD_SIZE = (1200, 800)
FPS        = 60
EDITOR_SIDEBAR_W = 160

WEIGHTS = {
    "separation": 2.0,
    "alignment":  1.0,
    "cohesion":   0.8,
    "relay":      0.5,
}

# ── signal strength ──────────────────────────────────────────────────────────
SIGNAL_RADIUS    = 260   # full signal inside this px distance to relay
SIGNAL_LOS_PEN   = 0.3   # multiplier when LOS is blocked
SIGNAL_TIER_HIGH = 0.6
SIGNAL_TIER_LOW  = 0.3

# ── airframe classes ─────────────────────────────────────────────────────────
# Used by renderer to pick shape and by config below for size defaults.
AIRFRAME = {
    "small":  {"shape": "circle",  "base_size": 3},
    "medium": {"shape": "diamond", "base_size": 5},
    "large":  {"shape": "triangle","base_size": 7},
}

# ── drone types ──────────────────────────────────────────────────────────────
DRONE_TYPES = {
    # key: (color, size-px, speed_mult, perception_mult, airframe)
    "fast":             {"color": (225, 235, 245), "size": 3, "speed_mult": 1.4, "perception_mult": 1.0, "airframe": "small"},
    "emp":              {"color": (175,  70, 220), "size": 3, "speed_mult": 1.2, "perception_mult": 1.0, "airframe": "small"},
    "jammer":           {"color": ( 35, 185, 185), "size": 3, "speed_mult": 1.0, "perception_mult": 1.0, "airframe": "small"},
    "interceptor_net":  {"color": ( 70, 215, 215), "size": 3, "speed_mult": 1.5, "perception_mult": 1.0, "airframe": "small"},
    "interceptor_fuse": {"color": ( 50, 195, 195), "size": 4, "speed_mult": 1.2, "perception_mult": 1.0, "airframe": "small"},
    "heavy":            {"color": (230, 145,  35), "size": 5, "speed_mult": 0.6, "perception_mult": 1.0, "airframe": "medium"},
    "loiter":           {"color": (215, 195,  35), "size": 5, "speed_mult": 0.7, "perception_mult": 1.0, "airframe": "medium"},
    "smokescreen":      {"color": (155, 160, 165), "size": 5, "speed_mult": 0.9, "perception_mult": 1.0, "airframe": "medium"},
    "decoy":            {"color": (195, 160,  25), "size": 5, "speed_mult": 0.8, "perception_mult": 1.0, "airframe": "medium"},
    "recon":            {"color": ( 75, 195,  75), "size": 7, "speed_mult": 1.1, "perception_mult": 1.5, "airframe": "large"},
    "relay_backup":     {"color": (175, 155,  45), "size": 7, "speed_mult": 0.8, "perception_mult": 1.0, "airframe": "large"},
}

# ── enemy drone types ────────────────────────────────────────────────────────
ENEMY_DRONE_TYPES = {
    "hunter":   {"color": (225,  55,  55), "size": 4, "speed_mult": 1.2},
    "kamikaze": {"color": (155,  25,  25), "size": 3, "speed_mult": 1.5},
}

# ── loadout ──────────────────────────────────────────────────────────────────
LOADOUT = {
    "fast":             0.30,
    "heavy":            0.10,
    "recon":            0.08,
    "interceptor_net":  0.08,
    "interceptor_fuse": 0.04,
    "loiter":           0.08,
    "smokescreen":      0.06,
    "jammer":           0.04,
    "emp":              0.04,
    "decoy":            0.06,
    "relay_backup":     0.12,
}

LOADOUT_PRESETS = [
    {   # Default
        "fast": 0.30, "heavy": 0.10, "recon": 0.08,
        "interceptor_net": 0.08, "interceptor_fuse": 0.04, "loiter": 0.08,
        "smokescreen": 0.06, "jammer": 0.04, "emp": 0.04,
        "decoy": 0.06, "relay_backup": 0.12,
    },
    {   # Assault
        "fast": 0.40, "heavy": 0.20, "recon": 0.05,
        "interceptor_net": 0.05, "interceptor_fuse": 0.05, "loiter": 0.05,
        "smokescreen": 0.04, "jammer": 0.02, "emp": 0.06,
        "decoy": 0.03, "relay_backup": 0.05,
    },
    {   # Defensive
        "fast": 0.18, "heavy": 0.05, "recon": 0.10,
        "interceptor_net": 0.15, "interceptor_fuse": 0.10, "loiter": 0.05,
        "smokescreen": 0.10, "jammer": 0.08, "emp": 0.02,
        "decoy": 0.05, "relay_backup": 0.12,
    },
]
LOADOUT_PRESET_IDX   = 0
LOADOUT_PRESET_NAMES = ["Default", "Assault", "Defensive"]

# ── mission phases ───────────────────────────────────────────────────────────
MISSION_PHASES = ["TRANSIT", "SUPPRESSION", "SATURATION", "PROSECUTION", "PERSISTENCE"]

PHASE_FORMATION = {
    "TRANSIT":     "COLUMN",
    "SUPPRESSION": "DENSE",
    "SATURATION":  "DENSE",
    "PROSECUTION": "PURSUIT",
    "PERSISTENCE": "DISPERSED",
}

# ── formation modes ──────────────────────────────────────────────────────────
FORMATION_MODES = ["COLUMN", "DENSE", "DISPERSED", "PURSUIT", "BUBBLE"]

# ── RECON ────────────────────────────────────────────────────────────────────
RECON_RANGE          = 400   # px scan radius
RECON_LEAD_DIST      = 150   # px ahead of relay
RECON_PROJ_ALERT     = 300   # px — projectile to relay triggers alert

# ── threats / turrets ────────────────────────────────────────────────────────
TURRET_FIRE_RATE  = 1.0    # shots/second
TURRET_RANGE      = 300.0  # px
PROJECTILE_SPEED  = 600.0  # px/second
PROJECTILE_RADIUS = 3
DRONE_HIT_RADIUS  = 5
RELAY_HIT_RADIUS  = 11
NOFLYZONE_KILLS   = True

# ── special weapon constants ─────────────────────────────────────────────────
JAMMER_RANGE       = 200    # px — turrets inside have miss chance
JAMMER_MISS_CHANCE = 0.50

EMP_LEAD_DIST      = 300    # px ahead of relay → EMP flies here
EMP_ARMING_DELAY   = 3.0    # seconds hovering before detonation
EMP_EFFECT_RADIUS  = 400    # px blast radius
EMP_DISABLE_TIME   = 15.0   # seconds turret is disabled
EMP_STUN_TIME      = 8.0    # seconds enemy loses tracking
EMP_BLAST_DURATION = 1.5    # seconds visual

SMOKESCREEN_RANGE    = 200  # px — detect incoming projectiles within
SMOKESCREEN_RADIUS   = 80   # px — cloud radius
SMOKESCREEN_MISS_CH  = 0.70 # miss chance through smoke
SMOKESCREEN_CHARGES  = 3    # max clouds per drone
SMOKESCREEN_CD       = 2.0  # seconds cooldown between deploys
SMOKESCREEN_PERSIST  = 6.0  # seconds cloud lasts

INTERCEPTOR_NET_RANGE  = 40  # px — trigger net deploy
INTERCEPTOR_NET_RADIUS = 60  # px — net effect radius
INTERCEPTOR_NET_CD     = 3.0 # seconds cooldown after deploy
INTERCEPTOR_FUSE_RANGE = 25  # px — trigger fuse detonation

BUBBLE_RADIUS = 80   # px around relay for BUBBLE formation

# ── relay succession ─────────────────────────────────────────────────────────
MIMICRY_DELAY_MIN = 5.0
MIMICRY_DELAY_MAX = 15.0

# ── convergence ──────────────────────────────────────────────────────────────
CONVERGENCE_ENABLED    = True
CONVERGENCE_T_LEAD     = 8.0   # seconds of lead time relay broadcasts T-zero

# ── reflex layer ─────────────────────────────────────────────────────────────
MIN_SAFE_DISTANCE = 8    # px — collision avoidance triggers below this
GEOFENCE_MARGIN   = 25   # px from boundary
GEOFENCE_FORCE    = 2.0  # force magnitude at boundary

# ── metrics ──────────────────────────────────────────────────────────────────
TARGET_REACH_FRACTION = 0.5
END_OVERLAY_SECONDS   = 4.0
CSV_LOG_PATH          = "logs/run_log.csv"

# ── enemy swarm defaults ─────────────────────────────────────────────────────
ENEMY_SWARM_SIZE         = 15
ENEMY_SWARM_COMPOSITION  = {"hunter": 0.6, "kamikaze": 0.4}
ENEMY_SPAWN_DELAY        = 10.0  # seconds
