import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import hashlib
import json
from tempfile import NamedTemporaryFile

# 配置参数
MODEL_PATH = "best.pt"
ALARM_SOUND = "alarm.mp3"
MAX_HISTORY = 5
CONF_THRESHOLD = 0.5
ALARM_DURATION = 1
USER_DATA_FILE = "users.json"

# 初始化用户数据
def initialize_users():
    if not os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "w") as f:
            json.dump({}, f)


def register_user(username, password):
    with open(USER_DATA_FILE, "r") as f:
        users = json.load(f)
    if username in users:
        return False
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    users[username] = {
        'salt': salt.hex(),
        'key': key.hex()
    }
    with open(USER_DATA_FILE, "w") as f:
        json.dump(users, f)
    return True


def verify_user(username, password):
    with open(USER_DATA_FILE, "r") as f:
        users = json.load(f)
    if username not in users:
        return False
    user = users[username]
    salt = bytes.fromhex(user['salt'])
    key = bytes.fromhex(user['key'])
    new_key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return key == new_key


# 初始化用户数据
initialize_users()


# 自定义样式
def set_custom_style():
    st.markdown(f"""
        <style>
            .main {{ background: #f8f9fa; }}
            .stAlert {{ border-radius: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .video-container {{
                position: relative;
                padding: 20px;
                background: white;
                border-radius: 15px;
                margin: 15px 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            }}
            .status-badge {{
                position: absolute;
                top: 25px;
                right: 25px;
                z-index: 100;
                padding: 8px 15px;
                border-radius: 20px;
                font-weight: bold;
            }}
            #alarm-audio {{ display: none; }}
            .auth-container {{ 
                max-width: 400px;
                margin: 2rem auto;
                padding: 2rem;
                background: white;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
        </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        st.stop()


def auth_component():
    set_custom_style()
    with st.container():
        st.markdown("<div class='auth-container'>", unsafe_allow_html=True)
        st.title("用户认证 🔐")

        if 'auth_view' not in st.session_state:
            st.session_state.auth_view = "登录"

        auth_mode = st.radio(
            "请选择操作",
            ["登录", "注册"],
            key="auth_radio",
            index=0 if st.session_state.auth_view == "登录" else 1
        )

        username = st.text_input("用户名", key="auth_username")
        password = st.text_input("密码", type="password", key="auth_password")

        if auth_mode == "注册":
            confirm_password = st.text_input("确认密码", type="password")
            if st.button("注册"):
                if password != confirm_password:
                    st.error("密码不一致")
                elif len(password) < 6:
                    st.error("密码至少需要6位")
                elif not username:
                    st.error("用户名不能为空")
                else:
                    if register_user(username, password):
                        st.success("注册成功，请登录")
                        st.session_state.auth_view = "登录"
                        st.rerun()
                    else:
                        st.error("用户名已存在")
        else:
            if st.button("登录"):
                if verify_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("用户名或密码错误")
        st.markdown("</div>", unsafe_allow_html=True)


def handle_audio_permission():
    """自动处理音频播放权限"""
    autoplay_script = """
    <script>
    (function() {
        const audioElement = document.getElementById('alarm-audio');

        // 自动静音初始化
        const initAudio = () => {
            audioElement.muted = true;
            audioElement.play()
                .then(() => {
                    audioElement.pause();
                    audioElement.muted = false;
                    window.audioInitialized = true;
                })
                .catch(error => console.log('自动初始化完成'));
        }

        // 首次加载初始化
        if(document.readyState === 'complete') {
            initAudio();
        } else {
            document.addEventListener('DOMContentLoaded', initAudio);
        }

        // 处理页面变化时的重新初始化
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) initAudio();
        });
    })();
    </script>
    """
    st.components.v1.html(autoplay_script)


def play_alarm():
    """触发警报声音"""
    play_script = """
    <script>
    (function() {
        const audio = document.getElementById('alarm-audio');
        if(window.audioInitialized) {
            audio.play().catch(error => {
                console.log('自动播放失败，尝试重新初始化');
                audio.muted = true;
                audio.play().then(() => {
                    audio.pause();
                    audio.muted = false;
                    audio.play();
                });
            });
        }
    })();
    </script>
    """
    st.components.v1.html(play_script, height=0)


def handle_image_detection(model):
    st.subheader("图片检测")
    uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"], key="image_uploader")

    if uploaded_file:
        if not os.path.exists(ALARM_SOUND):
            st.error(f"警报音频文件不存在: {ALARM_SOUND}")
            return

        start_time = time.time()
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("正在分析..."):
            results = model.predict(img, conf=CONF_THRESHOLD)
            annotated_img = results[0].plot()

            detected_classes = []
            for box in results[0].boxes:
                if box.conf > CONF_THRESHOLD:
                    detected_classes.append(model.names[int(box.cls)])

            fire_detected = any(cls in ['Fire', 'smoke'] for cls in detected_classes)
            process_time = time.time() - start_time

            record = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "has_Fire": fire_detected,
                "result": "发现危险" if fire_detected else "环境安全",
                "process_time": process_time,
                "type": "image"
            }
            st.session_state.history.append(record)

            col1, col2 = st.columns(2)
            with col1:
                st.image(img, channels="BGR", caption="原始图片")
            with col2:
                st.image(annotated_img, channels="BGR", caption="检测结果")

            if fire_detected:
                st.error("## 🚨 检测到火灾危险！")
                play_alarm()
            else:
                st.success("## ✅ 环境安全")


def handle_video_detection(model):
    st.subheader("视频检测")
    uploaded_file = st.file_uploader("上传视频", type=["mp4", "avi", "mov"], key="video_uploader")

    if uploaded_file and not st.session_state.video_processing:
        if not os.path.exists(ALARM_SOUND):
            st.error(f"警报音频文件不存在: {ALARM_SOUND}")
            return

        st.session_state.video_processing = True
        video_path = f"temp_{uploaded_file.name}"

        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        required_frames = max(1, int(fps * ALARM_DURATION))

        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        alarm_zone = st.empty()
        consecutive_danger_frames = 0
        alarm_triggered = False
        alarm_flag = False
        processed_frames = 0
        start_time = time.time()

        while st.session_state.video_processing and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=CONF_THRESHOLD)
            annotated_frame = results[0].plot()

            detected_classes = []
            for box in results[0].boxes:
                if box.conf > CONF_THRESHOLD:
                    detected_classes.append(model.names[int(box.cls)])

            fire_detected = any(cls in ['Fire', 'smoke'] for cls in detected_classes)

            if fire_detected:
                consecutive_danger_frames += 1
                duration = consecutive_danger_frames / fps

                if consecutive_danger_frames >= required_frames:
                    if not alarm_triggered:
                        play_alarm()
                        alarm_triggered = True
                        alarm_flag = True
                    alarm_zone.error(f"## 🚨 持续危险！已持续 {duration:.1f} 秒")
                else:
                    alarm_zone.warning(f"## ⚠️ 检测到危险！已持续 {duration:.1f} 秒")
            else:
                consecutive_danger_frames = 0
                alarm_triggered = False
                alarm_zone.empty()

            frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
            progress_bar.progress((processed_frames + 1) / total_frames)
            processed_frames += 1

        process_time = time.time() - start_time
        cap.release()
        st.session_state.video_processing = False

        if alarm_flag:
            st.error(f"## 🚨 检测到持续{ALARM_DURATION}秒的火灾危险！")
        else:
            st.success("## ✅ 视频分析完成，未发现持续危险")

        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "has_Fire": alarm_flag,
            "result": "发现危险" if alarm_flag else "环境安全",
            "process_time": process_time,
            "frames": processed_frames,
            "type": "video"
        }
        st.session_state.history.append(record)


def handle_realtime_detection(model):
    st.subheader("实时检测")

    # 云端环境提示
    if os.environ.get('IS_STREAMLIT_CLOUD', False):
        st.info("""
        💡 云端检测提示：
        1. 使用下方上传视频模拟实时检测
        2. 本地运行时仍可使用摄像头
        """)

        # 备用视频上传方案
        uploaded_file = st.file_uploader("上传测试视频", type=["mp4"])
        if uploaded_file:
            # 复用视频检测逻辑
            handle_video_detection(model)
        return

    # 本地摄像头检测
    run_camera = st.checkbox("开启摄像头")
    frame_placeholder = st.empty()
    alarm_zone = st.empty()
    status_bar = st.empty()

    danger_start_time = None
    alarm_triggered = False
    consecutive_danger_frames = 0

    if run_camera:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("无法访问摄像头设备")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            required_frames = max(1, int(fps * ALARM_DURATION))

            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    status_bar.error("无法获取摄像头画面")
                    break

                # 推理检测
                results = model.predict(frame, conf=CONF_THRESHOLD)
                annotated_frame = results[0].plot()

                # 分析结果
                detected_classes = [
                    model.names[int(box.cls)]
                    for box in results[0].boxes
                    if box.conf > CONF_THRESHOLD
                ]
                fire_detected = any(cls in ['Fire', 'smoke'] for cls in detected_classes)

                # 警报逻辑
                if fire_detected:
                    consecutive_danger_frames += 1
                    duration = consecutive_danger_frames / fps

                    if consecutive_danger_frames >= required_frames:
                        if not alarm_triggered:
                            play_alarm()
                            alarm_triggered = True
                        alarm_zone.error(f"## 🚨 持续危险！已持续 {duration:.1f} 秒")
                    else:
                        alarm_zone.warning(f"## ⚠️ 检测到危险！已持续 {duration:.1f} 秒")
                else:
                    consecutive_danger_frames = 0
                    alarm_triggered = False
                    alarm_zone.empty()

                # 显示画面
                frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

        except Exception as e:
            st.error(f"摄像头访问失败：{str(e)}")
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()


def main_app():
    st.set_page_config(
        page_title="基于YOLO的火灾检测系统设计与实现",
        page_icon="🔥",
        layout="wide"
    )
    set_custom_style()
    model = load_model()

    # 初始化session状态
    session_defaults = {
        "history": [],
        "video_processing": False
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # 音频元素和权限处理
    st.markdown(f'''
        <audio id="alarm-audio" controls style="display:none;">
            <source src="{ALARM_SOUND}" type="audio/mpeg">
        </audio>
    ''', unsafe_allow_html=True)
    handle_audio_permission()

    st.title("🔥 基于YOLO的火灾检测系统设计与实现")
    st.markdown("---")

    with st.sidebar:
        st.write(f"当前用户：{st.session_state.username}")
        if st.button("注销"):
            st.session_state.authenticated = False
            st.rerun()

        st.header("控制面板")
        detection_mode = st.radio(
            "检测模式",
            ["图片检测", "视频检测", "实时检测"],
            index=0
        )
        st.markdown("---")
        st.subheader("检测历史")
        for idx, item in enumerate(st.session_state.history[-MAX_HISTORY:]):
            status_color = "#dc3545" if item["has_Fire"] else "#28a745"
            st.markdown(f"""
                <div class="video-container">
                    <div class="status-badge" style="background: {status_color}; color: white;">
                        {item['result']}
                    </div>
                    <div class="stats">
                        时间: {item['time']}<br>
                        处理: {item['process_time']:.2f}s<br>
                        帧数: {item.get('frames', 1)}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        if st.button("清空历史"):
            st.session_state.history = []

    if detection_mode == "图片检测":
        handle_image_detection(model)
    elif detection_mode == "视频检测":
        handle_video_detection(model)
    elif detection_mode == "实时检测":
        handle_realtime_detection(model)


def main():
    if not hasattr(st.session_state, 'authenticated') or not st.session_state.authenticated:
        auth_component()
    else:
        main_app()

if __name__ == "__main__":
    main()