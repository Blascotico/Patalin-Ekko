import streamlit as st
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from datetime import datetime
import tempfile
from moviepy.editor import VideoFileClip


# ========================================================
# CONFIG GENERAL
# ========================================================
st.set_page_config(
    page_title="Ekko-WEB v0.0b",
    page_icon="👣",
    layout="wide"
)

if "vista" not in st.session_state:
    st.session_state.vista = "video"

if "df_result" not in st.session_state:
    st.session_state.df_result = None


# ========================================================
# CSS
# ========================================================
st.markdown("""
<style>
.stApp { background-color: #000000 !important; }
body { background-color: #000 !important; }

* { color: #e0e0e0 !important; font-family: 'Segoe UI', sans-serif; }

/* Cards */
.data-card {
    background: #111;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 0 10px #000 inset, 0 0 6px #000;
    margin-bottom: 16px;
}
.data-card h3 {
    margin: 0;
    font-size: 20px;
    color: #9ad0ff !important;
}
.data-value {
    font-size: 28px;
    font-weight: bold;
    margin-top: 8px;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)


# ========================================================
# Mediapipe
# ========================================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ========================================================
# Medición de pies
# ========================================================
def calcular_pies(landmarks, img_shape):
    h, w = img_shape[:2]

    try:
        Lh = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
        Lt = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]

        Rh = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]
        Rt = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

        okL = Lh.visibility > 0.5 and Lt.visibility > 0.5
        okR = Rh.visibility > 0.5 and Rt.visibility > 0.5

        if not (okL or okR):
            return None, None

        L_len = None
        R_len = None

        if okL:
            L_len = np.linalg.norm(
                np.array([Lt.x * w, Lt.y * h]) -
                np.array([Lh.x * w, Lh.y * h])
            )

        if okR:
            R_len = np.linalg.norm(
                np.array([Rt.x * w, Rt.y * h]) -
                np.array([Rh.x * w, Rh.y * h])
            )

        return L_len, R_len

    except:
        return None, None



# ========================================================
# PROCESAR VIDEO
# ========================================================
if st.session_state.vista == "video":

    st.title("👣 Patalin Ekko")
    st.write("Subí un video para analizarlo automáticamente.")

    uploaded_video = st.file_uploader("Elegir video:", type=["mp4", "mkv", "avi", "mov"])

    col_video, col_data = st.columns([2, 1])

    video_box = col_video.empty()
    progress_bar = col_video.progress(0)

    live_L = col_data.empty()
    live_R = col_data.empty()
    live_log = col_data.empty()

    if uploaded_video:

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        # Cargar video con MoviePy
        clip = VideoFileClip(video_path)
        fps = clip.fps
        total_frames = int(clip.duration * fps)

        results = []
        logs = ""

        for frame_id, frame in enumerate(clip.iter_frames(), start=1):

            progress_bar.progress(frame_id / total_frames)

            rgb = frame  # ya viene en RGB

            res = pose.process(rgb)

            Lcm, Rcm = None, None

            if res.pose_landmarks:
                L, R = calcular_pies(res.pose_landmarks.landmark, rgb.shape)

                if L and R:
                    px_to_cm = 25 / ((L + R) / 2)
                else:
                    px_to_cm = 0.1

                if L:
                    Lcm = L * px_to_cm
                if R:
                    Rcm = R * px_to_cm

                # Dibujar pose
                mp_drawing.draw_landmarks(rgb, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # ====================================================
            # Resize inteligente del video
            # ====================================================
            max_size = 500
            h, w, _ = rgb.shape
            aspect = w / h

            if w >= h:
                new_w = max_size
                new_h = int(new_w / aspect)
            else:
                new_h = max_size
                new_w = int(new_h * aspect)

            resized = cv2.resize(rgb, (new_w, new_h))
            video_box.image(resized, channels="RGB")

            # ====================================================
            #  Diseño de los datos
            # ====================================================
            live_L.markdown(
                f"""
                <div class="data-card">
                    <h3>Pie Izquierdo</h3>
                    <div class="data-value">{Lcm} cm</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            live_R.markdown(
                f"""
                <div class="data-card">
                    <h3>Pie Derecho</h3>
                    <div class="data-value">{Rcm} cm</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if frame_id % 20 == 0:
                logs += f"[{frame_id}] L={Lcm}  |  R={Rcm}\n"
                live_log.code(logs)

            results.append({
                "frame": frame_id,
                "time": frame_id / fps,
                "left_cm": Lcm,
                "right_cm": Rcm
            })

        st.session_state.df_result = pd.DataFrame(results)
        st.session_state.vista = "resultados"
        st.rerun()



# ========================================================
# RESULTADOS FINALES
# ========================================================
elif st.session_state.vista == "resultados":

    st.title("📊 Resultados del Análisis")

    df = st.session_state.df_result

    st.subheader("Resumen")

    c1, c2 = st.columns(2)

    c1.markdown(
        f"""
        <div class="data-card">
            <h3>Promedio Pie Izquierdo</h3>
            <div class="data-value">{df['left_cm'].mean():.2f} cm</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c2.markdown(
        f"""
        <div class="data-card">
            <h3>Promedio Pie Derecho</h3>
            <div class="data-value">{df['right_cm'].mean():.2f} cm</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Tabla completa")
    st.dataframe(df, use_container_width=True)

    st.subheader("Curvas de medición")
    st.line_chart(df[["left_cm", "right_cm"]])

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Descargar CSV",
        data=csv_data,
        file_name=f"patalin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    if st.button("🔄 Nuevo análisis"):
        st.session_state.vista = "video"
        st.rerun()
