import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(layout="wide")
st.title("Visualisasi Data Seismik")

# =========================
# UPLOAD DATA
# =========================
st.sidebar.header("Input Data")

files = st.sidebar.file_uploader(
    "Upload data seismik (.txt / .csv)",
    type=["txt", "csv"],
    accept_multiple_files=True
)

datasets = {}

if files:
    for file in files:
        if file.name.endswith(".csv"):
            data = np.loadtxt(file, delimiter=",")
        else:
            data = np.loadtxt(file)
        datasets[file.name] = data

    selected_file = st.sidebar.selectbox(
        "Pilih Dataset",
        list(datasets.keys())
    )
    data = datasets[selected_file]

else:
    st.warning("Belum ada file → digunakan data contoh")
    np.random.seed(0)
    nt, nx = 300, 40
    t = np.linspace(0, 2, nt)
    data = np.sin(10 * np.outer(t, np.ones(nx))) * np.exp(-t[:, None])
    data += 0.2 * np.random.randn(nt, nx)

nt, nx = data.shape
t = np.arange(nt)

# =========================
# PRE-PROCESSING
# =========================
st.sidebar.header("Pre-processing")

# Normalisasi
if st.sidebar.checkbox("Normalisasi Amplitudo"):
    data = data / np.max(np.abs(data))

# =========================
# FILTER MOVING AVERAGE (FIX ERROR)
# =========================
if st.sidebar.checkbox("Filter Moving Average"):
    window = st.sidebar.slider("Ukuran Window Filter", 3, 21, 5, step=2)

    # Pastikan window tidak lebih besar dari jumlah sampel
    if window > nt:
        st.warning("Ukuran window terlalu besar, disesuaikan otomatis.")
        window = nt - 1 if nt % 2 == 0 else nt

    kernel = np.ones(window) / window
    data_filt = np.zeros_like(data)

    for i in range(nx):
        conv = np.convolve(data[:, i], kernel, mode="same")

        # POTONG agar panjangnya sama dengan data
        start = (len(conv) - nt) // 2
        data_filt[:, i] = conv[start:start + nt]

    data = data_filt

# =========================
# KONTROL TRACE
# =========================
st.sidebar.header("Kontrol Trace")

trace_min, trace_max = st.sidebar.slider(
    "Rentang Trace",
    0, nx - 1,
    (0, nx - 1)
)

data = data[:, trace_min:trace_max + 1]
nx = data.shape[1]

# =========================
# KONTROL SKALA AMPLITUDO
# =========================
st.sidebar.header("Skala Amplitudo")

scale_mode = st.sidebar.radio(
    "Mode Skala",
    ["Auto Scale", "Manual Scale"]
)

vmin = vmax = None

if scale_mode == "Manual Scale":
    amp = float(np.max(np.abs(data)))
    vmin, vmax = st.sidebar.slider(
        "Atur vmin & vmax",
        -amp, amp,
        (-0.5 * amp, 0.5 * amp)
    )

# =========================
# OPSI TAMBAHAN
# =========================
reverse_time = st.sidebar.checkbox("Balik Sumbu Waktu")

# =========================
# PILIH TIPE PLOT
# =========================
plot_type = st.sidebar.selectbox(
    "Tipe Plot",
    ["Wiggle Trace", "Image Seismik"]
)

scale = st.sidebar.slider(
    "Skala Wiggle",
    0.1, 5.0, 1.0
)

# =========================
# FUNGSI WIGGLE TRACE
# =========================
def wiggle(ax, data, t, scale):
    nt, nx = data.shape
    for i in range(nx):
        trace = data[:, i] * scale
        ax.plot(i + trace, t, color="black", linewidth=0.8)
        ax.fill_betweenx(
            t, i, i + trace,
            where=(trace > 0),
            color="black"
        )

# =========================
# PLOT
# =========================
fig, ax = plt.subplots(figsize=(10, 6))

if plot_type == "Wiggle Trace":
    wiggle(ax, data, t, scale)
    ax.set_xlim(-1, nx + 1)

else:
    cmap = st.sidebar.selectbox(
        "Colormap",
        ["gray", "seismic", "viridis", "plasma"]
    )

    if scale_mode == "Auto Scale":
        im = ax.imshow(
            data,
            cmap=cmap,
            aspect="auto",
            extent=[0, nx, t.max(), t.min()]
        )
    else:
        im = ax.imshow(
            data,
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            extent=[0, nx, t.max(), t.min()]
        )

    plt.colorbar(im, ax=ax, label="Amplitudo")

if reverse_time:
    ax.invert_yaxis()

ax.set_xlabel("Trace")
ax.set_ylabel("Waktu")
ax.set_title("Penampang Seismik")

st.pyplot(fig)

# =========================
# SIMPAN GAMBAR
# =========================
buf = BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
buf.seek(0)

st.download_button(
    "💾 Simpan Plot sebagai Gambar",
    data=buf,
    file_name="penampang_seismik.png",
    mime="image/png"
)
