from openai import OpenAI
import time

# 初始化 LM Studio Qwen Model
client = OpenAI(base_url="your url", api_key="your api key")

# 定義所有節氣
SEASONS = [
    "立春", "雨水", "驚蟄", "春分", "清明", "穀雨",
    "立夏", "小滿", "芒種", "夏至", "小暑", "大暑",
    "立秋", "處暑", "白露", "秋分", "寒露", "霜降",
    "立冬", "小雪", "大雪", "冬至", "小寒", "大寒"
]

def generate_qa_batch(prompt, batch_size, season):
    """
    調用本地 LLM 生成一批特定節氣的問答集。

    Args:
        prompt (str): 用於生成問答集的提示。
        batch_size (int): 每次生成的問答組數量。
        season (str): 當前的節氣。

    Returns:
        str: LLM 生成的原始回應。
    """
    system_prompt = f"""
        你現在是一位專業且友善的台灣康是美藥妝店的智能導購小姐。
        你的服務對象主要是20-30歲的職場男性且目前正逢{season}的節氣。
        你的主要任務是針對這個年齡層的顧客在購買保健產品時可能遇到的選擇困難，
        提供專業的導購建議和產品推薦，以協助他們找到最適合的商品。
        你需要熟悉康是美店內常見的保健產品種類、品牌、成分、功效以及不同年齡層與節氣養生需求。
        請生成 {batch_size} 組包含顧客問題和你的專業回答的問答集，並按照以下格式呈現每一組問答：

        "[
        <SeasonType>節氣:{season}
        <Age>年齡:20-30
        <Gender>性別:男
        <Property>屬性:職場男性
        <Protype>商品屬性:保健產品
        <s>Human: {{問題}}
        <s>Assistant: {{回答}}
        ]"
    """

    response = client.chat.completions.create(
        model="gemma-3-12b-it",
        temperature=0.7,  # 可以根據需求調整
        max_tokens=8192 * (batch_size // 5),  # 增加最大 token 數以容納更多問答
        top_p=0.95,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"請生成 {batch_size} 組問答集。"}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    total_qa_per_season = 5  # 每個節氣生成的問答組數量
    batch_size = 5
    delay_seconds = 2  # 增加延遲，避免過於頻繁的請求

    all_season_responses = {}

    print(f"開始生成每個節氣 {total_qa_per_season} 組問答集...")

    for season in SEASONS:
        print(f"\n--- 開始生成 {season} 的問答集 ---")
        all_responses_for_season = []
        num_batches = total_qa_per_season // batch_size

        for i in range(num_batches):
            #print(f"\n--- 開始生成第 {i * batch_size + 1} - {(i + 1) * batch_size} 組問答集 ({season}) ---")
            prompt = f"請生成接下來的 {batch_size} 組問答集。"
            llm_response = generate_qa_batch(prompt, batch_size, season)
            print("本批 LLM 原始回應:\n", llm_response)
            all_responses_for_season.append(llm_response)

            if i < num_batches - 1:
                print(f"\n--- 本批生成完畢，休息 {delay_seconds} 秒... ---")
                time.sleep(delay_seconds)

        # 處理剩餘不足一個批次的問答
        remaining = total_qa_per_season % batch_size
        if remaining > 0:
            print(f"\n--- 開始生成剩餘的 {remaining} 組問答集 ({season}) ---")
            prompt = f"請生成接下來的 {remaining} 組問答集。"
            llm_response = generate_qa_batch(prompt, remaining, season)
            print("本批 LLM 原始回應:\n", llm_response)
            all_responses_for_season.append(llm_response)
            print(f"\n--- 本批生成完畢，休息 {delay_seconds} 秒... ---")
            time.sleep(delay_seconds)

        # 將當前節氣的所有回應合併
        final_response_for_season = "\n".join(all_responses_for_season)
        all_season_responses[season] = final_response_for_season

    # 將所有節氣的回應儲存到不同的檔案或一個總檔案
    output_prefix = "qa_dataset"
    try:
        for season, response in all_season_responses.items():
            output_filename = f"{output_prefix}_{season}.txt"
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(response)
            print(f"\n{season} 的 {total_qa_per_season} 組問答集已成功儲存至 {output_filename}")

        # 可選：將所有節氣的問答集合併到一個總檔案
        all_in_one_filename = f"{output_prefix}_all_seasons.txt"
        with open(all_in_one_filename, "w", encoding="utf-8") as f:
            for season, response in all_season_responses.items():
                f.write(f"--- {season} ---\n")
                f.write(response)
                f.write("\n\n")
            print(f"\n所有節氣的問答集已合併儲存至 {all_in_one_filename}")

    except Exception as e:
        print(f"\n儲存檔案時發生錯誤: {e}")