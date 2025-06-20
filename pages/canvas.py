import streamlit as st
import uuid
from PIL import Image
import io
import json
import base64
import time
import requests
from streamlit_drawable_canvas import st_canvas # Đảm bảo đã import thư viện này

# --- Hàm hỗ trợ (đã định nghĩa trước đó) ---
# Dán hàm add_circle_to_json_data và get_gaze_coordinates_from_llm vào đây
# (Không cần thay đổi các hàm này)

def add_circle_to_json_data(current_json_data, x, y, radius, color, stroke_width):
    """
    Thêm một đối tượng hình tròn vào dữ liệu JSON của canvas.
    `x` và `y` là tọa độ tâm. Fabric.js sử dụng tọa độ góc trên bên trái để định vị đối tượng.
    """
    if current_json_data is None or not current_json_data:
        current_json_data = {"objects": []}

    if isinstance(current_json_data, str):
        current_json_data = json.loads(current_json_data)

    current_json_data_copy = json.loads(json.dumps(current_json_data))

    circle_object = {
        "type": "circle",
        "originX": "left",
        "originY": "top",
        "left": x - radius,
        "top": y - radius,
        "radius": radius,
        "width": radius * 2,
        "height": radius * 2,
        "fill": f"{color}55",
        "stroke": color,
        "strokeWidth": stroke_width,
        "selectable": True,
        "evented": True,
    }
    new_objects = [obj.copy() if isinstance(obj, dict) else obj for obj in current_json_data_copy.get("objects", [])]
    new_objects.append(circle_object)
    return {"objects": new_objects}


def get_gaze_coordinates_from_llm(image_bytes, canvas_width, canvas_height):
    base64_image_data = base64.b64encode(image_bytes).decode('utf-8')

    prompt = (
        f"Phân tích hình ảnh người được cung cấp. Xác định tâm ước tính của ánh mắt trên hình ảnh. "
        f"Phản hồi bằng một đối tượng JSON duy nhất chứa 'gaze_x' và 'gaze_y' làm tọa độ trong hình ảnh, "
        f"giả sử góc trên bên trái là (0,0) và góc dưới bên phải là ({canvas_width},{canvas_height}). "
        "Ví dụ: {\"gaze_x\": 350, \"gaze_y\": 250}. Hãy chính xác và chỉ xuất ra JSON."
    )
    use_mock_response = False
    apiKey = ""
    try:
        api_key_local = st.secrets.get("GOOGLE_API_KEY", "")
        if not api_key_local:
            raise ValueError("Không tìm thấy khóa API trong st.secrets. Đang mô phỏng phản hồi.")
        apiKey = api_key_local
    except Exception as e:
        use_mock_response = True
        st.warning(f"Lưu ý: {e}. Đang sử dụng phản hồi LLM mô phỏng để phát hiện ánh mắt.")

    if use_mock_response:
        mock_gaze_x = int(canvas_width / 2 + (st.session_state.get('mock_gaze_offset_x', 0) % 100 - 50))
        mock_gaze_y = int(canvas_height / 2 + (st.session_state.get('mock_gaze_offset_y', 0) % 100 - 50))
        st.session_state['mock_gaze_offset_x'] = (st.session_state.get('mock_gaze_offset_x', 0) + 10) % 100
        st.session_state['mock_gaze_offset_y'] = (st.session_state.get('mock_gaze_offset_y', 0) + 15) % 100
        time.sleep(1)
        return {"gaze_x": max(0, min(canvas_width, mock_gaze_x)), "gaze_y": max(0, min(canvas_height, mock_gaze_y))}
    else:
        try:
            apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": base64_image_data
                                }
                            }
                        ]
                    }
                ]
            }
            response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, json=payload, timeout=30).json()

            try:
                response_json = response
                llm_gaze_text = response_json['candidates'][0]['content']['parts'][0]['text']
                gaze_coords = json.loads(llm_gaze_text)
                return gaze_coords
            except (KeyError, IndexError) as parse_error:
                st.error(f"❌ Định dạng phản hồi Gemini không mong muốn: {parse_error}")
            except json.JSONDecodeError as json_error:
                st.error(f"❌ Lỗi giải mã JSON: {json_error}\nVăn bản thô: {llm_gaze_text}")
            except Exception as general_error:
                st.error(f"❌ Lỗi không mong muốn: {general_error}")
            return {"gaze_x": -1, "gaze_y": -1}
        except requests.exceptions.RequestException as req_e:
            st.error(f"Lỗi khi gọi API LLM: {req_e}. Vui lòng kiểm tra khóa API và kết nối mạng của bạn.")
            return {"gaze_x": -1, "gaze_y": -1}
        except Exception as api_e:
            st.error(f"Đã xảy ra lỗi không mong muốn trong quá trình gọi API: {api_e}")
            return {"gaze_x": -1, "gaze_y": -1}


def canvas():
    # --- Cấu hình Trang Streamlit ---
    # ĐÃ XÓA: st.set_page_config(layout="wide", page_title="Simple Drawing App", page_icon=":pencil2:")
    # Điều này nên nằm trong tệp app.py chính của bạn, nếu bạn muốn cấu hình trang toàn cầu.

    # --- Khởi tạo Trạng thái Phiên ---
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = str(uuid.uuid4())
    if 'drawing_history' not in st.session_state:
        st.session_state.drawing_history = []
    if 'history_index' not in st.session_state:
        st.session_state.history_index = -1
    if 'action_triggered_rerun' not in st.session_state:
        st.session_state.action_triggered_rerun = False
    if 'expected_canvas_json_str' not in st.session_state:
        st.session_state.expected_canvas_json_str = json.dumps({"objects": []}, sort_keys=True)
    if 'captured_image_bytes' not in st.session_state:
        st.session_state.captured_image_bytes = None
    if 'last_captured_image_for_display' not in st.session_state:
        st.session_state.last_captured_image_for_display = None

    # Khởi tạo trạng thái selectbox cho chế độ vẽ
    if 'drawing_mode_value' not in st.session_state:
        st.session_state.drawing_mode_value = "freedraw" # Đặt giá trị mặc định ban đầu


    st.title("Canvas Vẽ Tương tác với Phát hiện Ánh mắt AI (Mô phỏng)")

    # --- Cấu hình Thanh bên (Sidebar) ---
    st.sidebar.header("Cấu hình Canvas")

    drawing_mode_options = ("freedraw", "line", "rect", "circle", "transform", "polygon", "point")
    selected_index = 0
    try:
        selected_index = drawing_mode_options.index(st.session_state.drawing_mode_value)
    except ValueError:
        st.session_state.drawing_mode_value = "freedraw"
        selected_index = 0

    drawing_mode = st.sidebar.selectbox(
        "Công cụ vẽ:",
        drawing_mode_options,
        index=selected_index,
        key="drawing_tool_selectbox"
    )

    if drawing_mode != st.session_state.drawing_mode_value:
        st.session_state.drawing_mode_value = drawing_mode

    stroke_width = st.sidebar.slider("Độ dày nét vẽ:", 1, 25, 3)

    point_display_radius = 0
    if drawing_mode == "point":
        point_display_radius = st.sidebar.slider("Bán kính hiển thị điểm:", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Màu nét vẽ (hex):", "#000000")
    bg_color = st.sidebar.color_picker("Màu nền (hex):", "#FFFFFF")

    bg_image_file = st.sidebar.file_uploader("Tải lên ảnh nền:", type=["png", "jpg", "jpeg"])
    
    bg_image = None
    if bg_image_file:
        try:
            bg_image = Image.open(bg_image_file)
        except Exception as e:
            st.sidebar.error(f"Lỗi khi tải ảnh: {e}. Vui lòng thử tệp khác.")

    initial_canvas_drawing_data = {"objects": []}
    if st.session_state.history_index >= 0 and st.session_state.history_index < len(st.session_state.drawing_history):
        initial_canvas_drawing_data = json.loads(json.dumps(st.session_state.drawing_history[st.session_state.history_index]))

    st.session_state.expected_canvas_json_str = json.dumps(initial_canvas_drawing_data, sort_keys=True)

    CANVAS_HEIGHT = 500
    CANVAS_WIDTH = 700

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=bg_image,
        update_streamlit=True,
        height=CANVAS_HEIGHT,
        width=CANVAS_WIDTH,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius,
        display_toolbar=False,
        key=st.session_state.canvas_key,
        initial_drawing=initial_canvas_drawing_data
    )

    if canvas_result.json_data is not None:
        current_json_str_from_canvas = json.dumps(canvas_result.json_data, sort_keys=True)

        if not st.session_state.action_triggered_rerun and \
           st.session_state.expected_canvas_json_str != current_json_str_from_canvas:

            if st.session_state.history_index < len(st.session_state.drawing_history) - 1:
                st.session_state.drawing_history = st.session_state.drawing_history[:st.session_state.history_index + 1]

            st.session_state.drawing_history.append(json.loads(json.dumps(canvas_result.json_data)))
            st.session_state.history_index = len(st.session_state.drawing_history) - 1
            st.session_state.expected_canvas_json_str = current_json_str_from_canvas
            
            st.rerun()
    
    st.session_state.action_triggered_rerun = False

    undo_disabled = st.session_state.history_index <= 0
    redo_disabled = st.session_state.history_index >= len(st.session_state.drawing_history) - 1

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Xóa Canvas", use_container_width=True, key="clear_canvas_button"):
            st.session_state.canvas_key = str(uuid.uuid4())
            st.session_state.drawing_history = []
            st.session_state.history_index = -1
            st.session_state.expected_canvas_json_str = json.dumps({"objects": []}, sort_keys=True)
            st.session_state.action_triggered_rerun = True
            st.session_state.drawing_mode_value = "freedraw"
            st.rerun()
    with col2:
        if st.button("Hoàn tác", use_container_width=True, disabled=undo_disabled, key="undo_button"):
            if st.session_state.history_index > 0:
                st.session_state.history_index -= 1
                st.session_state.action_triggered_rerun = True
                st.rerun()
    with col3:
        if st.button("Làm lại", use_container_width=True, disabled=redo_disabled, key="redo_button"):
            if st.session_state.history_index < len(st.session_state.drawing_history) - 1:
                st.session_state.history_index += 1
                st.session_state.action_triggered_rerun = True
                st.rerun()
    with col4:
        if canvas_result.image_data is not None:
            pil_image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Tải ảnh",
                data=byte_im,
                file_name="drawn_image.png",
                mime="image/png",
                use_container_width=True,
                key="download_button_active"
            )
        else:
            st.button("Tải ảnh", disabled=True, use_container_width=True, key="download_button_disabled")

    st.markdown("---")

    st.header("Phát hiện Ánh mắt AI (Mô phỏng)")
    st.write("Tính năng này **mô phỏng** việc phát hiện ánh mắt bằng cách sử dụng mô hình AI (LLM). Để phát hiện ánh mắt thực tế, chính xác, theo thời gian thực, thường yêu cầu các hệ thống thị giác máy tính chuyên biệt, điều này vượt quá phạm vi của một ứng dụng Streamlit cơ bản.")

    camera_input = st.camera_input("Chụp một khung hình từ webcam của bạn")

    if camera_input:
        st.session_state.captured_image_bytes = camera_input.getvalue()
        st.session_state.last_captured_image_for_display = camera_input
        st.image(camera_input, caption="Khung hình đã chụp", width=300)

    if st.button("Phân tích Ánh mắt từ Khung hình đã chụp", disabled=(st.session_state.captured_image_bytes is None)):
        if st.session_state.captured_image_bytes:
            with st.spinner("Đang phân tích ánh mắt..."):
                gaze_coords = get_gaze_coordinates_from_llm(
                    st.session_state.captured_image_bytes,
                    CANVAS_WIDTH, CANVAS_HEIGHT
                )
                gaze_x = int(gaze_coords.get('gaze_x', -1))
                gaze_y = int(gaze_coords.get('gaze_y', -1))

                if gaze_x != -1 and gaze_y != -1:
                    st.success(f"AI gợi ý điểm nhìn là: ({gaze_x}, {gaze_y})")

                    current_state_for_drawing = initial_canvas_drawing_data
                    
                    updated_json_data = add_circle_to_json_data(
                        current_state_for_drawing,
                        gaze_x, gaze_y,
                        radius=10,
                        color="#FF0000",
                        stroke_width=2
                    )
                    
                    st.session_state.drawing_history = st.session_state.drawing_history[:st.session_state.history_index + 1]
                    st.session_state.drawing_history.append(json.loads(json.dumps(updated_json_data)))
                    st.session_state.history_index = len(st.session_state.drawing_history) - 1
                    
                    st.session_state.action_triggered_rerun = True
                    st.rerun()
                else:
                    st.error("AI không thể xác định tọa độ ánh mắt. Vui lòng thử lại.")
        else:
            st.warning("Vui lòng chụp ảnh trước bằng cách sử dụng đầu vào camera.")

    if st.session_state.last_captured_image_for_display:
        st.image(st.session_state.last_captured_image_for_display, caption="Khung hình cuối cùng đã chụp để phân tích", width=300)