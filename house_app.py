"""House Price Prototype - Google Earth Engine + Machine Learning  (single file, v2)

Run:   just press the Play button (or: python house_app.py). It starts itself.
       Manual:  python -m streamlit run house_app.py

Self-checking: at start-up every satellite layer is tested on your city. If a dataset is
unavailable, the app tries a backup source, and if that fails too it simply drops that
layer (the model handles missing features) instead of crashing.

Demo mode  = real satellite data + SYNTHETIC prices (to test the pipeline).
Real mode  = upload a listings CSV: lat, lon, area_sqft, price [, bhk].
"""
import importlib.util
import os
import subprocess
import sys

# --- Bootstrap: install missing packages, and start Streamlit automatically ---------
_REQUIRED = {"streamlit": "streamlit", "streamlit_folium": "streamlit-folium",
             "folium": "folium", "ee": "earthengine-api", "sklearn": "scikit-learn",
             "pandas": "pandas", "numpy": "numpy", "joblib": "joblib"}


def _has(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ValueError, ImportError):
        return mod in sys.modules


_missing = [pkg for mod, pkg in _REQUIRED.items() if not _has(mod)]
if _missing:
    print("Installing missing packages:", ", ".join(_missing), flush=True)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *_missing])
    except Exception:
        sys.exit("Could not install: " + ", ".join(_missing) + "\n"
                 f"Run:  {sys.executable} -m pip install " + " ".join(_missing))

import streamlit as st  # noqa: E402
from streamlit.runtime.scriptrunner import get_script_run_ctx  # noqa: E402

try:
    _in_streamlit = get_script_run_ctx(suppress_warning=True) is not None
except TypeError:
    _in_streamlit = get_script_run_ctx() is not None
if not _in_streamlit:  # started with Play / "python house_app.py" -> relaunch properly
    print("Starting the app in your browser (keep this window open, Ctrl+C to stop)...", flush=True)
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__),
                        "--browser.gatherUsageStats=false"])
    except KeyboardInterrupt:
        pass
    sys.exit(0)

import ee  # noqa: E402
import folium  # noqa: E402
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GroupKFold
from streamlit_folium import st_folium

DEFAULT_PROJECT = "house-price-510108"
BUFFER_M = 500
Y0, Y1 = "2025-01-01", "2025-12-31"
TAG = "v2"
CITIES = {"Pune": (18.5204, 73.8567), "Nagpur": (21.1458, 79.0882),
          "Mumbai": (19.0760, 72.8777), "Bengaluru": (12.9716, 77.5946),
          "Hyderabad": (17.3850, 78.4867), "Delhi": (28.6139, 77.2090)}
LABELS = {"ndvi": "Greenery (NDVI)", "ndbi": "Built-up index (NDBI)", "ndwi": "Water index (NDWI)",
          "night_lights": "Night lights", "elevation": "Elevation (m)", "slope": "Slope (deg)",
          "pop_density": "Population density", "built_prob": "Built-up probability",
          "water_dist_m": "Distance to water (m)"}
FEATURES = list(LABELS)
MODEL_FEATURES = FEATURES + ["dist_center_km", "log_area", "bhk"]


# ------------------------- satellite layers (with backups) -------------------
def _indices(im):
    return ee.Image.cat([im.normalizedDifference(["B8", "B4"]).rename("ndvi"),
                         im.normalizedDifference(["B11", "B8"]).rename("ndbi"),
                         im.normalizedDifference(["B3", "B8"]).rename("ndwi")])


def _s2(aoi, d0, d1, cloud):
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi)
           .filterDate(d0, d1).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud))
           .select(["B3", "B4", "B8", "B11"]))
    return _indices(col.median())


def _dem(name):
    dem = ee.Image(name).select("elevation")
    return ee.Image.cat([dem.rename("elevation"), ee.Terrain.slope(dem).rename("slope")])


def _dw(aoi, d0, d1):
    return (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(aoi).filterDate(d0, d1)
            .select("built").mean().rename("built_prob"))


def _water(aoi):
    occ = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gt(50).selfMask()
    return occ.distance(ee.Kernel.euclidean(5000, "meters")).unmask(5000).rename("water_dist_m")


# group -> (title, bands, [candidate builders tried in order])
GROUPS = {
    "s2": ("Sentinel-2 optical", ["ndvi", "ndbi", "ndwi"], [
        lambda a: _s2(a, Y0, Y1, 20),
        lambda a: _s2(a, "2023-01-01", Y1, 40)]),
    "night": ("Night lights (VIIRS)", ["night_lights"], [
        lambda a: ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG").select("avg_rad")
        .sort("system:time_start", False).limit(12).median().rename("night_lights"),
        lambda a: ee.Image(ee.ImageCollection("NOAA/VIIRS/DNB/ANNUAL_V21").select("average")
                           .sort("system:time_start", False).first()).rename("night_lights")]),
    "terrain": ("Elevation & slope", ["elevation", "slope"], [
        lambda a: _dem("USGS/SRTMGL1_003"),
        lambda a: _dem("NASA/NASADEM_HGT/001")]),
    "pop": ("Population density", ["pop_density"], [
        lambda a: ee.Image(ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density")
                           .sort("system:time_start", False).first())
        .select("population_density").rename("pop_density"),
        lambda a: ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.eq("year", 2020))
        .mosaic().select("population").rename("pop_density")]),
    "built": ("Built-up (Dynamic World)", ["built_prob"], [
        lambda a: _dw(a, Y0, Y1),
        lambda a: _dw(a, "2023-01-01", Y1)]),
    "water": ("Surface water", ["water_dist_m"], [_water]),
}


@st.cache_resource
def gee_init(project):
    key = os.environ.get("EE_SERVICE_ACCOUNT_JSON")
    if key:  # hosted deployment
        import json
        info = json.loads(key)
        ee.Initialize(ee.ServiceAccountCredentials(info["client_email"], key_data=key),
                      project=project)
        return True
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    return True


@st.cache_resource(show_spinner=False)
def probe_layers(project, center):
    """Test every layer on the city centre; return {group: candidate_index}, {group: error}."""
    pt = ee.Geometry.Point([center[1], center[0]])
    aoi, chosen, notes = pt.buffer(30000), {}, {}
    for group, (_, _, cands) in GROUPS.items():
        for i, build in enumerate(cands):
            try:
                vals = build(aoi).reduceRegion(ee.Reducer.mean(), pt.buffer(BUFFER_M), 30).getInfo()
                if vals and all(v is not None for v in vals.values()):
                    chosen[group] = i
                    break
                notes[group] = "returned no data"
            except Exception as e:  # dataset missing / empty / renamed
                notes[group] = str(e).splitlines()[0][:110]
    return chosen, notes


def build_stack(chosen, aoi):
    return ee.Image.cat([GROUPS[g][2][i](aoi) for g, i in chosen.items()])


def extract(df, chosen, aoi, progress=None, chunk=50):
    """Add satellite FEATURES (NaN where a layer is unavailable) to df[lat, lon]."""
    df = df.reset_index(drop=True)
    stack, rows = build_stack(chosen, aoi), []
    for s in range(0, len(df), chunk):
        feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]).buffer(BUFFER_M),
                            {"idx": int(i)}) for i, r in df.iloc[s:s + chunk].iterrows()]
        res = stack.reduceRegions(ee.FeatureCollection(feats), ee.Reducer.mean(),
                                  scale=30).getInfo()
        for f in res["features"]:
            p = f["properties"]
            row = {"idx": p["idx"]}
            for k in FEATURES:
                v = p.get(k)
                row[k] = np.nan if v is None else v
            rows.append(row)
        if progress:
            progress.progress(min(1.0, (s + chunk) / len(df)))
    out = df.join(pd.DataFrame(rows).set_index("idx"))
    out[FEATURES] = out[FEATURES].astype(float)
    return out


@st.cache_data(show_spinner=False)
def satellite_profile(lat, lon, chosen_items):
    aoi = ee.Geometry.Point([lon, lat]).buffer(5000)
    return extract(pd.DataFrame({"lat": [lat], "lon": [lon]}), dict(chosen_items),
                   aoi).iloc[0][FEATURES].astype(float)


# ------------------------------- model --------------------------------------
def haversine_km(lat, lon, lat0, lon0):
    lat, lon, lat0, lon0 = map(np.radians, (lat, lon, lat0, lon0))
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def model_frame(df, center):
    d = df.copy()
    d["dist_center_km"] = haversine_km(d["lat"], d["lon"], *center)
    d["log_area"] = np.log(d["area_sqft"])
    if "bhk" not in d:
        d["bhk"] = np.nan
    return d


def make_demo(center, n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"lat": center[0] + rng.uniform(-0.12, 0.12, n),
                       "lon": center[1] + rng.uniform(-0.12, 0.12, n),
                       "area_sqft": rng.integers(450, 2500, n)})
    df["bhk"] = np.clip((df["area_sqft"] / 550).round(), 1, 5)
    return df


def synth_price(df, center, seed=0):
    """SYNTHETIC prices from real satellite features (pipeline test only)."""
    rng = np.random.default_rng(seed)
    d = haversine_km(df["lat"], df["lon"], *center)
    ppsf = (9000 - 220 * d + 6000 * df["ndvi"].fillna(0.2)
            + 250 * np.log1p(df["night_lights"].fillna(0).clip(lower=0))
            - 1.5 * df["water_dist_m"].fillna(5000) / 100)
    return (ppsf.clip(3500) * df["area_sqft"] * rng.normal(1, 0.08, len(df))).round(-3)


def train(df, center):
    d = model_frame(df.dropna(subset=["price", "area_sqft"]), center)
    feats = [f for f in MODEL_FEATURES if d[f].notna().any()]  # drop empty layers
    X, y = d[feats], np.log(d["price"])
    mk = lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                               min_samples_leaf=8, random_state=0)
    groups = d["lat"].round(2).astype(str) + "_" + d["lon"].round(2).astype(str)
    k, mape, r2 = min(5, groups.nunique()), None, None
    if k >= 2:  # spatial CV: whole map cells are held out
        pred = np.zeros(len(d))
        for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
            pred[te] = mk().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        mape = float(mean_absolute_percentage_error(d["price"], np.exp(pred)))
        r2 = float(r2_score(y, pred))
    return {"model": mk().fit(X, y), "features": feats, "center": center, "n": len(d),
            "mape": mape, "r2": r2, "means": d[FEATURES].mean(),
            "stds": d[FEATURES].std().replace(0, 1).fillna(1),
            "max_dist": float(d["dist_center_km"].max())}


# ---------------------------------- UI --------------------------------------
st.set_page_config(page_title="GEE House Price Prototype", layout="wide")
st.title("🏠 Satellite-powered House Price Estimator")

with st.sidebar:
    project = st.text_input("Earth Engine project ID", DEFAULT_PROJECT)
    city = st.selectbox("City", list(CITIES))
    source = st.radio("Training data", ["Demo (synthetic prices)", "My listings CSV (real prices)"])
    upload = st.file_uploader("Listings CSV", type="csv") if source.startswith("My") else None
    train_btn = st.button("Train on my CSV") if upload is not None else False

try:
    gee_init(project)
except Exception as e:
    st.error(f"Could not connect to Earth Engine: {e}")
    st.info("Check the project is registered at code.earthengine.google.com/register "
            "and the Earth Engine API is enabled in Google Cloud Console.")
    st.stop()

center = CITIES[city]
with st.spinner(f"Checking satellite datasets for {city} (first time only)..."):
    chosen, notes = probe_layers(project, center)
if not chosen:
    st.error("No satellite dataset could be read. Details: " + "; ".join(notes.values()))
    st.stop()
with st.sidebar.expander("Satellite layers", expanded=bool(notes and len(chosen) < len(GROUPS))):
    for g, (title, _, _) in GROUPS.items():
        if g in chosen:
            st.write(f"✅ {title}" + (" (backup source)" if chosen[g] else ""))
        else:
            st.write(f"⚠️ {title} unavailable, skipped")

is_demo = source.startswith("Demo")
path = f"model_{TAG}_{'demo' if is_demo else 'real'}_{city}.joblib"
if is_demo and st.sidebar.button("Rebuild demo model") and os.path.exists(path):
    os.remove(path)

try:
    if train_btn:
        data = pd.read_csv(upload)
        missing = {"lat", "lon", "area_sqft", "price"} - set(data.columns)
        if missing:
            st.error(f"CSV is missing columns: {sorted(missing)}")
            st.stop()
        data = data.dropna(subset=["lat", "lon", "area_sqft", "price"])
        aoi = ee.Geometry.Rectangle([float(data.lon.min()) - .05, float(data.lat.min()) - .05,
                                     float(data.lon.max()) + .05, float(data.lat.max()) + .05])
        bar = st.progress(0.0, "Reading satellite data for your listings...")
        joblib.dump(train(extract(data, chosen, aoi, bar), center), path)
        bar.empty()
    elif is_demo and not os.path.exists(path):
        bar = st.progress(0.0, f"Building demo model for {city} from real satellite data (~1-2 min)...")
        d = extract(make_demo(center), chosen,
                    ee.Geometry.Point([center[1], center[0]]).buffer(30000), bar)
        d["price"] = synth_price(d, center)
        joblib.dump(train(d, center), path)
        bar.empty()
except Exception as e:
    st.error(f"Earth Engine request failed: {str(e).splitlines()[0][:300]}")
    st.info("Press R to retry. If it repeats, send me this message.")
    st.stop()

if not os.path.exists(path):
    st.info("Upload a listings CSV (columns: lat, lon, area_sqft, price, optional bhk) "
            "and press **Train on my CSV**.")
    st.stop()
bundle = joblib.load(path)

if is_demo:
    st.warning("DEMO MODE: prices are synthetic, so estimates are illustrative only. "
               "Train on a real listings CSV for meaningful numbers.")
if bundle["mape"] is not None:
    st.caption(f"Trained on {bundle['n']} points · spatial-CV error ≈ {bundle['mape']:.0%} "
               f"(R² {bundle['r2']:.2f})")

key = f"pt_{city}"
st.session_state.setdefault(key, center)
lat, lon = st.session_state[key]

left, right = st.columns([3, 2])
with left:
    st.caption("Click the map to choose the property location")
    m = folium.Map(location=center, zoom_start=11)
    folium.Marker([lat, lon], tooltip="Selected property").add_to(m)
    click = st_folium(m, height=430, width=None, key=f"map_{city}",
                      returned_objects=["last_clicked"])
    if click and click.get("last_clicked"):
        new = (click["last_clicked"]["lat"], click["last_clicked"]["lng"])
        if new != st.session_state[key]:
            st.session_state[key] = new
            st.rerun()
with right:
    st.write(f"**Selected:** {lat:.5f}, {lon:.5f}")
    area = st.number_input("Area (sq ft)", 200, 10000, 1000, 50)
    bhk = st.selectbox("BHK", [1, 2, 3, 4, 5], index=1)
    go = st.button("Estimate price", type="primary")

if go:
    try:
        with st.spinner("Reading satellite data from Earth Engine..."):
            prof = satellite_profile(round(lat, 4), round(lon, 4), tuple(sorted(chosen.items())))
    except Exception as e:
        st.error(f"Earth Engine request failed: {str(e).splitlines()[0][:300]}")
        st.stop()
    row = model_frame(pd.DataFrame([{"lat": lat, "lon": lon, "area_sqft": area, "bhk": bhk,
                                     **prof}]), center)
    price = float(np.exp(bundle["model"].predict(row[bundle["features"]])[0]))
    err = bundle["mape"] or 0.15
    if row["dist_center_km"].iloc[0] > bundle["max_dist"] * 1.25:
        st.warning("This location is outside the area the model was trained on; treat it with caution.")
    a, b, c = st.columns(3)
    a.metric("Estimated price", f"₹{price / 1e5:,.1f} lakh")
    b.metric("Likely range", f"₹{price * (1 - err) / 1e5:,.0f} – {price * (1 + err) / 1e5:,.0f} lakh")
    c.metric("Price per sq ft", f"₹{price / area:,.0f}")

    st.subheader("Neighbourhood profile from satellites (500 m)")
    z = ((prof - bundle["means"]) / bundle["stds"]).clip(-3, 3).fillna(0)
    t1, t2 = st.columns(2)
    t1.dataframe(pd.DataFrame({"Value": prof.round(3), "City avg": bundle["means"].round(3)})
                 .rename(index=LABELS), width="stretch")
    t2.bar_chart(z.rename(index=LABELS).rename("vs city average (std devs)"))
