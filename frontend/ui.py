import os
import gradio as gr
from gradio_modal import Modal
from datetime import datetime
import time
from pathlib import Path
import threading
import shutil
import uuid

from backend.rag_service import process_query, _format_user_info
from backend.speech_service import speech_to_text, text_to_speech

# Example prompts for the UI
example_prompts = [
    ["我想找一些抗老保養品，預算在2000元以內，有什麼推薦嗎？"],
    ["最近容易失眠，有什麼改善睡眠品質的保健食品建議嗎？"],
    ["我是一個上班族，想找一些提升免疫力的保健品，請推薦。"],
    ["請推薦一些適合更年期女性使用的保健食品。"],
    ["有什麼美白保養品適合敏感肌使用？"]
]

# LaTeX delimiters for proper math rendering
latex_delimiters = [
    {"left": "$", "right": "$", "display": True},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False}
]

SESSION_STATE = {}


def clean_user_info(user_info):
    """清理用戶信息中的emoji和特殊字符"""
    if not user_info:
        return {}

    cleaned_info = {}
    for key, value in user_info.items():
        if value and isinstance(value, str):
            # 移除emoji前綴
            cleaned_value = value.split(' ', 1)[-1] if ' ' in value else value
            cleaned_info[key] = cleaned_value
        else:
            cleaned_info[key] = value

    return cleaned_info


def respond(message, history):
    """Process user input and generate a response using the backend"""
    try:
        # Create user context information
        user_info = clean_user_info(SESSION_STATE.get("user_info", {}))

        context = {
            "user_info": user_info,
            "user_info_str": _format_user_info(user_info) if user_info else ""
        }

        print(f"用戶信息: {user_info}")

        # Process user history
        processed_history = []
        for val in history:
            if val[0] and val[1]:
                user_msg = val[0]
                # Extract assistant response without the thinking part
                assistant_msg = val[1].split(
                    "**End thinking**")[-1].strip() if "**End thinking**" in val[1] else val[1]
                processed_history.append({"role": "user", "content": user_msg})
                processed_history.append(
                    {"role": "assistant", "content": assistant_msg})

        full_response = ""
        for chunk in process_query(message, processed_history, context):
            full_response += chunk

            # 檢查是否包含"End thinking"標記
            if "**End thinking**" in full_response:
                display_response = full_response.split(
                    "**End thinking**")[-1].strip()

                # 生成音頻
                audio_path = text_to_speech(display_response)

                return [
                    display_response,
                    gr.Audio(value=audio_path, visible=False, autoplay=True,
                             elem_id="auto-audio-player", elem_classes="mini-audio-player")
                ]

        return full_response

        # test
        # display_response = message
        # audio_path = text_to_speech(display_response)
        # return [
        #     display_response,
        #     gr.Audio(value=audio_path, visible=False, autoplay=True, elem_id="auto-audio-player", elem_classes="mini-audio-player")
        # ]
    except Exception as e:
        raise gr.Error(f"發生錯誤: {str(e)}")


def create_ui():
    """Create and return the Gradio UI"""
    with gr.Blocks(theme="default", title="保養導購機器人") as demo:

        with gr.Row():
            with gr.Column(scale=10):
                gr.Markdown("## 🧬 保養導購機器人")
                gr.Markdown("您的專業保健顧問，提供客製化的保健食品和保養品推薦。")
            with gr.Column(scale=1, elem_id="avatar-col"):
                settings_btn = gr.Button(
                    value="", elem_classes="settings-btn-icon")

        chat_interface = gr.ChatInterface(
            respond,
            examples=example_prompts,
            chatbot=gr.Chatbot(latex_delimiters=latex_delimiters, scale=9, avatar_images=(
                None, "assets/bot_avatar.png"), type="messages"),
            type="messages"
        )

        with gr.Row(elem_id="mic-container", visible=True):
            mic_btn = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label=None,
                elem_id="mic-button",
                elem_classes="mic-btn",
                visible=True,
                format="wav"
            )

        # Welcome modal
        with Modal(visible=True) as welcome_modal:
            gr.Markdown("# 歡迎使用保健保養品智慧導購助手! 👋")
            gr.Markdown("我是您的專業購物顧問，可以為您：")
            gr.Markdown("- 根據需求推薦合適的保健食品")
            gr.Markdown("- 提供專業的保養品使用建議")
            gr.Markdown("- 解答您的各種美容保健問題")
            gr.Markdown("- 分析產品成分與功效")

            with gr.Row():
                start_explore_btn = gr.Button(
                    "立即開始", elem_classes="start-explore-btn")
                start_question_btn = gr.Button(
                    "客製化推薦", elem_classes="start-question-btn")

        # User info modal for customized recommendations
        with Modal(visible=False) as user_info_modal:
            gr.Markdown("## 幫助我們更了解您")

            # Gender selection
            gender_radio = gr.Radio(
                ["👨 男", "👩 女", "🌈 其他"],
                label="您的性別",
                value=None
            )

            # Zodiac sign selection
            zodiac_radio = gr.Radio(
                ["♈ 白羊座", "♉ 金牛座", "♊ 雙子座", "♋ 巨蟹座",
                 "♌ 獅子座", "♍ 處女座", "♎ 天秤座", "♏ 天蠍座",
                 "♐ 射手座", "♑ 摩羯座", "♒ 水瓶座", "♓ 雙魚座"],
                label="您的星座",
                value=None
            )

            # Age range selection
            age_radio = gr.Radio(
                ["18-24", "25-34", "35-44", "45+"],
                label="您的年齡範圍",
                value=None
            )

            # MBTI selection
            mbti_radio = gr.Radio(
                ["INFP", "ENFP", "INFJ", "ENFJ",
                 "INTJ", "ENTJ", "INTP", "ENTP",
                 "ISFP", "ESFP", "ISFJ", "ESFJ",
                 "ISTP", "ESTP", "ISTJ", "ESTJ"],
                label="您的MBTI類型",
                value=None
            )

            # Submit button
            submit_info_btn = gr.Button("完成設定", elem_classes="submit-info-btn")

        def process_audio(audio_file):
            if audio_file is None:
                return None

            try:
                print(f"嘗試讀取音頻檔案: {audio_file}")

                # 轉換為文本
                text = speech_to_text(audio_file)

                print(f"語音識別結果: '{text}'")

                if text and text.strip() and not text.startswith("speech to text error"):
                    return text
                else:
                    return "無法識別語音，請重試"
            except Exception as e:
                print(f"處理音頻時出錯: {str(e)}")
                import traceback
                traceback.print_exc()
                return f"處理音頻時出錯: {str(e)}"

        # Event bindings
        start_explore_btn.click(
            fn=lambda: Modal(visible=False),
            inputs=None,
            outputs=welcome_modal
        )

        mic_btn.stop_recording(
            fn=process_audio,
            inputs=[mic_btn],
            outputs=[chat_interface.textbox]
        )

        def show_user_info_modal():
            return [Modal(visible=False), Modal(visible=True)]

        start_question_btn.click(
            show_user_info_modal,
            None,
            [welcome_modal, user_info_modal]
        )

        settings_btn.click(
            fn=lambda: Modal(visible=True),
            inputs=None,
            outputs=user_info_modal
        )

        def save_user_info(gender, zodiac, age, mbti):
            global SESSION_STATE
            user_info = {
                "gender": gender,
                "zodiac": zodiac,
                "age": age,
                "mbti": mbti
            }

            SESSION_STATE["user_info"] = user_info
            gr.Success("完成設定！我們會根據您的個人特質，為您推薦最適合的產品建議 💝")
            return Modal(visible=False)

        submit_info_btn.click(
            fn=save_user_info,
            inputs=[gender_radio, zodiac_radio, age_radio, mbti_radio],
            outputs=[user_info_modal]
        )

        # Add CSS styles
        gr.HTML("""
        <style>
            /* 立即開始按鈕 - 使用綠色表示開始行動 */
            .start-explore-btn {
                background: linear-gradient(135deg, #00b09b, #96c93d) !important;
                color: white !important;
                border: none !important;
                padding: 10px 20px !important;
                border-radius: 8px !important;
                font-weight: bold !important;
                font-size: 16px !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
            }
            .start-explore-btn:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
                background: linear-gradient(135deg, #02c3ab, #a7da44) !important;
            }

            /* 客製化推薦按鈕 - 使用紫色表示個性化服務 */
            .start-question-btn {
                background: linear-gradient(135deg, #8e2de2, #4a00e0) !important;
                color: white !important;
                border: none !important;
                padding: 10px 20px !important;
                border-radius: 8px !important;
                font-weight: bold !important;
                font-size: 16px !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
            }
            .start-question-btn:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
                background: linear-gradient(135deg, #9f43f3, #5a17f1) !important;
            }

            /* 完成設定按鈕 - 使用藍色表示確認提交 */
            .submit-info-btn {
                background: linear-gradient(135deg, #2193b0, #6dd5ed) !important;
                color: white !important;
                border: none !important;
                padding: 10px 20px !important;
                border-radius: 8px !important;
                font-weight: bold !important;
                font-size: 16px !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
                width: 100% !important;
                margin-top: 20px !important;
            }
            .submit-info-btn:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
                background: linear-gradient(135deg, #25a4c4, #7de4fc) !important;
            }

            /* 設定按鈕 - 使用灰色表示設定功能 */
            .settings-btn-icon {
                background-color: transparent !important;
                background-image: url('https://img.icons8.com/ios-filled/50/ffffff/settings.png') !important;
                background-repeat: no-repeat !important;
                background-position: center !important;
                background-size: 24px 24px !important;
                border: none !important;
                border-radius: 50% !important;
                width: 36px !important;
                height: 36px !important;
                position: fixed !important;
                top: 12px !important;
                right: 12px !important;
                z-index: 1 !important;
                cursor: pointer !important;
                transition: background-color 0.2s ease;
            }
            .settings-btn-icon:hover {
                background-color: rgba(255, 255, 255, 0.1) !important;
            }
            audio, .gradio-audio, div:has(> audio), #auto-audio-player, .mini-audio-player, 
            *[class*="audio"], *[id*="audio"], audio[autoplay], [data-testid*="audio"] {
                display: none !important;
                visibility: hidden !important;
                position: absolute !important;
                left: -9999px !important;
                height: 0 !important;
                width: 0 !important;
                opacity: 0 !important;
                z-index: -1000 !important;
                overflow: hidden !important;
                clip: rect(0, 0, 0, 0) !important;
                pointer-events: none !important;
                max-height: 0 !important;
                max-width: 0 !important;
            }
        </style>
        """)

    return demo
