import os
import re
import requests
import zipfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import google.generativeai as genai
from collections import defaultdict
from konlpy.tag import Okt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# ✅ 모델 zip 다운로드 & 압축 해제
ZIP_URL = "https://drive.google.com/uc?export=download&id=1fup5me_ftaDLjHKuyfSNqBAINPN7KABc"
ZIP_PATH = "models/models_bundle.zip"
EXTRACT_DIR = "models/"

def download_and_extract_zip():
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR, exist_ok=True)

    if not os.path.exists("models/art_classification_model.keras"):
        print("📦 모델 ZIP 다운로드 중...")
        response = requests.get(ZIP_URL)
        with open(ZIP_PATH, "wb") as f:
            f.write(response.content)
        print("✅ 다운로드 완료")

        print("🧩 압축 해제 중...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("✅ 압축 해제 완료")

# 📥 다운로드 + 압축해제 먼저 실행
download_and_extract_zip()

# ✅ 모델 로드
school_labels = [
    '르네상스', '바로크', '로코코', '신고전주의', '낭만주의',
    '자연주의', '사실주의', '인상주의', '입체파&추상화'
]

school_model = load_model("models/art_classification_model.keras")

binary_models = {}
for school in school_labels:
    path = f"models/{school}_이진분류.keras"
    if os.path.exists(path):
        binary_models[school] = load_model(path)

# ✅ FastAPI 앱 생성
app = FastAPI()
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔐 Gemini API 설정
genai.configure(api_key="AIzaSyAGNPBS6pzxMbPUbHlSdfhX5rrthgDy9ko")
okt = Okt()

# ✅ 간단한 라우트 예시
@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse("""
    <html>
    <head><title>AI 미술 분석</title></head>
    <body>
      <h2>🎨 AI 미술 분석 시스템에 오신 걸 환영합니다!</h2>
      <form action="/upload" enctype="multipart/form-data" method="post">
        <input name="file" type="file">
        <input type="submit">
      </form>
    </body>
    </html>
    """)

# ✅ Render 포트 인식을 위한 uvicorn 실행
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")

## 🔹 3️⃣ 학파별 어휘사전 (리스트 형식, 최신 class_mapping 반영)
structured_vocab = {
    '르네상스': {
        '구성': ['원근법', '균형', '조화', '수학적 비례', '대칭'],
        '기법': ['선 원근법', '공기 원근법(스푸마토)', '명암법(키아로스쿠로)', '해부학적 접근'],
        '색감': ['따뜻한 색조', '명암 대비', '부드러운 색조 변화'],
        '주제': ['인체 중심', '자연', '신화', '종교'],
        '대표 예술가': ['레오나르도 다빈치', '미켈란젤로', '라파엘로', '도나텔로', '뒤러', '반에이크', '브뤼헐']
    },
    '바로크': {
        '구성': ['역동적 구도', '대각선 구도', '비대칭 균형', '과장된 움직임', '연극적 장면 구성'],
        '기법': ['키아로스쿠로', '테네브리즘', '강한 명암 대비', '극적 조명', '사실적 세부 묘사', '감각적 표현'],
        '색감': ['풍부한 색채', '강렬한 명암 대비', '금빛 강조', '어두운 배경과 강한 빛의 대비'],
        '주제': ['왕권의 신성함', '극적 감정 표현', '종교적 환희', '역사적 사건', '연극적 장면', '감각적 경험'],
        '대표 예술가': ['카라바조', '루벤스', '렘브란트', '벨라스케스', '베르니니', '페르메이르', '할스']
    },
    '로코코': {
        '구성': ['곡선 구도', '섬세한 균형', '부드러운 흐름', '우아한 화면 구성'],
        '기법': ['가벼운 붓 터치', '부드러운 색감', '섬세한 장식적 표현', '풍속화 기법', '초상화 기법'],
        '색감': ['파스텔 색조', '따뜻한 색감', '은은한 명암', '감각적 색채 배치'],
        '주제': ['귀족 문화', '유희와 쾌락', '세련된 감각', '풍속적 장면', '낭만적 분위기'],
        '대표 예술가': ['와토', '부셰', '프라고나르', '고야', '샤르댕', '레이놀즈', '게인즈버러']
    },
    '신고전주의': {
        '구성': ['엄격한 비례', '대칭적 구도', '명확한 윤곽', '정적인 화면 질서'],
        '기법': ['정확한 선묘', '매끄러운 붓 터치', '형태의 명확성', '해부학적 접근', '조각적 인체 표현'],
        '색감': ['절제된 색채', '균형 잡힌 명암', '선명한 색상', '차분한 분위기'],
        '주제': ['고대 신화', '역사적 사건', '이성적 질서', '도덕적 교훈', '고전적 이상미'],
        '대표 예술가': ['다비드', '앵그르', '부게로']
    },
    '낭만주의': {
        '구성': ['역동적 구도', '비대칭 균형', '감정적 흐름', '연극적 장면 구성'],
        '기법': ['자유로운 붓 터치', '명암의 강한 대비', '선명한 색채', '극적 조명', '표현적 기법'],
        '색감': ['강렬한 색감', '극적인 명암 대비', '감정적 색채 표현', '루벤스풍 색채 활용'],
        '주제': ['민족적 정체성', '감정적 해방', '혁명과 자유', '극적 사건', '자연과 초월적 감성'],
        '대표 예술가': ['들라크루아', '제리코', '터너', '고야', '프리드리히']
    },
    '자연주의': {
        '구성': ['사실적 구도', '조화로운 자연 배치', '균형 잡힌 풍경 표현'],
        '기법': ['야외 사생', '사실적 묘사', '섬세한 붓 터치', '자연 채광 활용'],
        '색감': ['자연스러운 색감', '부드러운 명암', '따뜻한 토속적 색조', '사실적인 빛 표현'],
        '주제': ['전원 풍경', '자연 속 삶', '소박한 농민의 일상', '자연의 생명력'],
        '대표 예술가': ['밀레', '코로', '터너', '컨스터블']
    },
    '사실주의': {
        '구성': ['균형 잡힌 구도', '일상적 장면 중심'],
        '기법': ['정밀한 묘사', '명확한 윤곽선', '자연스러운 명암 표현'],
        '색감': ['차분한 색감', '자연스러운 명암', '현실적인 색채 사용'],
        '주제': ['노동자', '평범한 시민', '사회 현실', '산업화', '빈부 격차'],
        '대표 예술가': ['쿠르베', '도미에', '밀레', '코로', '호머', '휘슬러', '호퍼']
    },
    '인상주의': {
        '구성': ['개방적인 구도', '특정 순간의 포착', '자연스러운 시선 이동 유도'],
        '기법': ['짧고 분할된 붓 터치', '색의 병치', '빛과 대기 표현 강조'],
        '색감': ['밝고 순수한 색채', '빛과 대기의 흐름에 따른 색상 변주'],
        '주제': ['일상적 장면', '도시와 자연 풍경', '순간적인 빛과 색채의 변화 포착'],
        '대표 예술가': ['마네', '모네', '르누아르', '드가', '피사로', '카유보트']
    },
    '입체파&추상화': {
        '구성': ['다중 시점', '원근법 파괴', '기하학적 구성', '화면의 평면화'],
        '기법': ['대상 분해 및 재구성', '점·선·면을 활용한 조형적 표현', '콜라주 기법'],
        '색감': ['단순한 색채', '원색 강조(차가운 추상)', '감성적 색면 표현(뜨거운 추상)'],
        '주제': ['본질 탐구', '재현의 거부', '감성과 직관의 표현', '기하학적 질서'],
        '대표 예술가': ['피카소', '브라크', '레제', '칸딘스키', '몬드리안', '말레비치', '로스코']
    }
}


# 🔍 Gemini 호출 함수

def get_gemini_keywords(image_path):
    img = Image.open(image_path).convert("RGB")
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = """이 그림은 어떤 미술 사조에 속하는지 분석하고 다음 형식으로 정리해줘.
    각 항목마다 핵심 단어(키워드)를 먼저 제시하고, 이어서 해당 설명을 작성해줘.
    형식:
    구성 - 핵심 단어1, 단어2, 단어3 : 설명
    기법 - 핵심 단어1, 단어2 : 설명
    색감 - 핵심 단어1, 단어2 : 설명
    주제 - 핵심 단어1, 단어2 : 설명
    """
    response = model.generate_content([prompt, img])
    return response.text

def extract_school_from_gemini(text):
    for school in structured_vocab:
        if school in text:
            return school
    return "미상"

def get_similarity_scores(image_path):
    img = Image.open(image_path).convert("RGB")
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = """
다음 중 이 그림에 가장 가까운 하나의 미술 사조를 아래 목록에서 선택하고, 형식에 맞춰 유사도를 출력하세요.
아래 목록 중 **하나만** 선택하세요:

- 르네상스
- 바로크
- 로코코
- 신고전주의
- 낭만주의
- 자연주의
- 사실주의
- 인상주의
- 입체파&추상화

형식 예시 (학파와 확률만 보여주세요):
르네상스: 92%
"""
    response = model.generate_content([prompt, img])
    return response.text.strip()

def extract_school_from_similarity_text(similarity_text):
    for school in structured_vocab:
        if school in similarity_text:
            return school
    return list(structured_vocab.keys())[0]  # fallback: 첫 번째 학파라도 매칭

def extract_similarity_score(similarity_text, label):
    """
    예시: similarity_text = "낭만주의: 88%"이고 label = "낭만주의"일 때 → "88%" 추출
    """
    pattern = rf"{label}\s*:\s*(\d+)%"
    match = re.search(pattern, similarity_text)
    if match:
        return f"{match.group(1)}%"
    return "유사도 0.5% 이하입니다."

# ✅ 이미지 전처리

def preprocess_image_for_model(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x

# ✅ Keras 다중분류

def predict_school_by_keras(img_path):
    x = preprocess_image_for_model(img_path)
    pred = school_model.predict(x)[0]
    return school_labels[np.argmax(pred)]

# ✅ 이진분류 유사도

def predict_similarity_for_school(img_path, predicted_school):
    model = binary_models.get(predicted_school)
    if not model:
        return 0.0
    x = preprocess_image_for_model(img_path)
    score = model.predict(x)[0][0]  # 확률값
    return round(score * 100, 2)

# ✅ 앙상블 분석

def hybrid_art_style_analysis(image_path):

    # Gemini 분석
    gemini_keywords = get_gemini_keywords(image_path)
    gemini_school = extract_school_from_gemini(gemini_keywords)

    gemini_similarity_text = get_similarity_scores(image_path)
    gemini_similarity_school = extract_school_from_similarity_text(gemini_similarity_text)
    gemini_similarity_score = extract_similarity_score(gemini_similarity_text, gemini_similarity_school)

    # Keras 분석
    keras_school = predict_school_by_keras(image_path)
    keras_similarity = predict_similarity_for_school(image_path, keras_school)

    # 최종 학파 결정 (투표식)
    votes = defaultdict(int)
    votes[gemini_school] += 1
    votes[gemini_similarity_school] += 1
    votes[keras_school] += 1
    final_school = max(votes.items(), key=lambda x: x[1])[0]

    # 유사도 평균
    try:
        gemini_score = int(gemini_similarity_score.replace('%', ''))
    except:
        gemini_score = 50

    final_similarity = round((gemini_score + keras_similarity) / 2)

    return {
        "최종 학파": final_school,
        "최종 유사도": f"{final_similarity}%",
        "Gemini 설명": gemini_keywords,
        "Keras 예측 학파": keras_school,
        "Gemini 예측 학파": gemini_similarity_school,
        "Keras 유사도": f"{keras_similarity}%"
    }

# 🔍 어휘 유사도 체크 (80% 이상 형태소 매칭)
def generate_summary_table(description, school):
    if school not in structured_vocab:
        return "관련 어휘와 일치하는 설명이 없습니다."
    import re
    from collections import defaultdict
    from konlpy.tag import Okt
    category_colors = {
        "구성": "#e74c3c",
        "기법": "#3498db",
        "색감": "#f1c40f",
        "주제": "#2ecc71"
    }
    okt = Okt()
    sentences = re.split(r"\n+", description.strip())
    final_html = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # 별표 및 카테고리 패턴 제거
        sentence = re.sub(r"\*\*+", "", sentence)
        # "색감 -", "구성 -" 등의 패턴 제거
        sentence = re.sub(r"\*\*(구성|기법|색감|주제)\s*-\s*[^:]+:\*\*", "", sentence)

        if not sentence:
            continue

        # 키워드 추출
        desc_morphs = okt.morphs(sentence)
        matched = defaultdict(set)
        for cat, vocab_list in structured_vocab[school].items():
            if cat == "대표 예술가":  # 대표 예술가 카테고리는 건너뛰기
                continue
            for word in vocab_list:
                word_morphs = okt.morphs(word)
                if any(w in desc_morphs for w in word_morphs):
                    matched[cat].add(word)

        # 표 생성
        table_html = "<table style='width:100%; border-collapse: collapse; margin-bottom:0.5rem;'>"
        table_html += "<tr style='background:#f9f9f9;'>"
        for cat in ["구성", "기법", "색감", "주제"]:
            table_html += f"<th style='padding:4px; border:1px solid #ccc'>{cat}</th>"
        table_html += "</tr><tr>"
        for cat in ["구성", "기법", "색감", "주제"]:
            kw_list = matched.get(cat, [])
            colored = [f"<b style='color:{category_colors[cat]}'>{kw}</b>" for kw in kw_list]
            table_html += f"<td style='padding:4px; border:1px solid #ccc'>{', '.join(colored) if colored else '-'}</td>"
        table_html += "</tr></table>"
        final_html += table_html
        final_html += f"<p style='margin-bottom:1.2rem'>📝 {sentence}</p>"

    return final_html or "관련 어휘와 일치하는 설명이 없습니다."



    result_html = ""
    for desc, matched_keywords in unique_descriptions.items():
        # 설명문 내부에 색 강조 적용
        highlighted_desc = highlight_keywords(desc, matched_keywords)

        result_html += f"<p>📝 <strong>설명:</strong> {highlighted_desc}</p>"

        result_html += """
        <table style='width:100%; border-collapse: collapse; margin-bottom: 1rem;'>
          <tr>
            <th style='border: 1px solid #ccc; padding: 6px;'>구성</th>
            <th style='border: 1px solid #ccc; padding: 6px;'>기법</th>
          </tr>
          <tr>
            <td style='border: 1px solid #ccc; padding: 6px;'>""" + \
                       ', '.join(matched_keywords.get("구성", ["-"])) + "</td>" + \
                       "<td style='border: 1px solid #ccc; padding: 6px;'>" + \
                       ', '.join(matched_keywords.get("기법", ["-"])) + "</td>" + \
                       "</tr><tr>" + \
                       "<th style='border: 1px solid #ccc; padding: 6px;'>색감</th>" + \
                       "<th style='border: 1px solid #ccc; padding: 6px;'>주제</th>" + \
                       "</tr><tr>" + \
                       "<td style='border: 1px solid #ccc; padding: 6px;'>" + \
                       ', '.join(matched_keywords.get("색감", ["-"])) + "</td>" + \
                       "<td style='border: 1px solid #ccc; padding: 6px;'>" + \
                       ', '.join(matched_keywords.get("주제", ["-"])) + "</td>" + \
                       "</tr></table>"

    return result_html

def highlight_keywords(text, keyword_categories):
    color_map = {
        "구성": "#e74c3c",    # 빨강
        "기법": "#3498db",    # 파랑
        "색감": "#f39c12",    # 노랑
        "주제": "#27ae60",    # 초록
    }

    for category, keywords in keyword_categories.items():
        for keyword in sorted(keywords, key=lambda x: -len(x)):  # 긴 키워드 먼저
            color = color_map.get(category, "#000")
            # 중복 강조 방지
            pattern = re.compile(re.escape(keyword))
            text = pattern.sub(
                f"<b style='color:{color}'>{keyword}</b>", text
            )
    return text
def generate_summary_table(description, school_vocab):
    okt = Okt()
    sentences = re.split(r"[.?!]\s*", description.strip())
    results = defaultdict(lambda: {"설명": [], "구성": set(), "기법": set(), "색감": set(), "주제": set()})

    for sentence in sentences:
        clean_sentence = sentence.strip()
        if not clean_sentence:
            continue
        morphs = okt.morphs(clean_sentence)
        for category, keywords in school_vocab.items():
            if category == "대표 예술가":
                continue
            for keyword in keywords:
                key_morphs = okt.morphs(keyword)
                match_count = sum(1 for m in key_morphs if m in morphs)
                match_ratio = match_count / len(key_morphs) if key_morphs else 0
                if match_ratio >= 0.8 or keyword in clean_sentence:
                    results[clean_sentence]["설명"].append(clean_sentence)
                    results[clean_sentence][category].add(keyword)

    color_map = {
        "구성": "#e74c3c",    # 빨강
        "기법": "#3498db",    # 파랑
        "색감": "#f39c12",    # 노랑
        "주제": "#27ae60",    # 초록
    }

    # 유일한 키워드 조합 단위로 병합
    group_map = {}
    for sent, data in results.items():
        key = tuple(sorted(data["구성"])) + tuple(sorted(data["기법"])) + tuple(sorted(data["색감"])) + tuple(sorted(data["주제"]))
        if key not in group_map:
            group_map[key] = {"설명들": [], "구성": data["구성"], "기법": data["기법"], "색감": data["색감"], "주제": data["주제"]}
        group_map[key]["설명들"].append(sent)

    final_html = ""

    for group in group_map.values():
        # 1행 4열 표 생성
        table_html = f"""
        <table style="width: 100%; text-align: center; border-collapse: collapse; margin-bottom: 0.5rem;">
          <tr>
            <th style="background-color: #ffe3e3;">구성</th>
            <th style="background-color: #d0ebff;">기법</th>
            <th style="background-color: #fff3bf;">색감</th>
            <th style="background-color: #d3f9d8;">주제</th>
          </tr>
          <tr>
            <td>{', '.join(group['구성']) if group['구성'] else '-'}</td>
            <td>{', '.join(group['기법']) if group['기법'] else '-'}</td>
            <td>{', '.join(group['색감']) if group['색감'] else '-'}</td>
            <td>{', '.join(group['주제']) if group['주제'] else '-'}</td>
          </tr>
        </table>
        """

        # 설명문 처리 및 키워드 강조
        descs = ""
        for s in group["설명들"]:
            colored_text = s
            for cat in ["구성", "기법", "색감", "주제"]:
                for kw in group[cat]:
                    colored_text = colored_text.replace(kw, f"<b style='color: {color_map[cat]}'>{kw}</b>")
            descs += f"<p style='margin: 0.3rem 0;'>{colored_text}</p>"

        final_html += table_html + descs

    return final_html if final_html else "관련 어휘와 일치하는 설명이 없습니다."

def render_summary_table(summary_dict):
    html_output += """
    <table style="
        border-collapse: collapse;
        margin: 1em 0;
        width: 100%;
        max-width: 600px;
        table-layout: fixed;
        background-color: #f9f9f9;
        border: 1px solid #ccc;
    ">
      <tr>
        <th style="width: 50%; padding: 10px; background-color: #eee; text-align: center;">구성</th>
        <th style="width: 50%; padding: 10px; background-color: #eee; text-align: center;">기법</th>
      </tr>
      <tr>
        <td style="padding: 10px; text-align: center; vertical-align: middle;">{구성}</td>
        <td style="padding: 10px; text-align: center; vertical-align: middle;">{기법}</td>
      </tr>
      <tr>
        <th style="width: 50%; padding: 10px; background-color: #eee; text-align: center;">색감</th>
        <th style="width: 50%; padding: 10px; background-color: #eee; text-align: center;">주제</th>
      </tr>
      <tr>
        <td style="padding: 10px; text-align: center; vertical-align: middle;">{색감}</td>
        <td style="padding: 10px; text-align: center; vertical-align: middle;">{주제}</td>
      </tr>
    </table>
    """.format(
        구성=", ".join(keyword_dict.get("구성", [])) or "-",
        기법=", ".join(keyword_dict.get("기법", [])) or "-",
        색감=", ".join(keyword_dict.get("색감", [])) or "-",
        주제=", ".join(keyword_dict.get("주제", [])) or "-"
    )

    return html_output
def format_gemini_text(gemini_text):
    import re

    color_map = {
        "구성": "#e74c3c",    # 빨강
        "기법": "#3498db",    # 파랑
        "색감": "#f39c12",    # 노랑
        "주제": "#27ae60",    # 초록
    }

    # 각 카테고리별 키워드 파싱
    sections = re.split(r"\n+", gemini_text.strip())
    category_keywords = {"구성": [], "기법": [], "색감": [], "주제": []}

    for sec in sections:
        match = re.match(r"(구성|기법|색감|주제)\s*-\s*(.*?):", sec)
        if match:
            cat, keywords_str = match.groups()
            keywords = [kw.strip() for kw in keywords_str.split(",")]
            category_keywords[cat].extend(keywords)

    # HTML 출력
    html = ""
    for cat in ["구성", "기법", "색감", "주제"]:
        if category_keywords[cat]:
            html += f"""
            <div style="background-color: {color_map[cat]}; color: white; font-weight: bold; padding: 0.5rem 1rem; border-radius: 6px; margin-bottom: 0.3rem;">
              {cat}
            </div>
            <div style="padding: 0.3rem 1rem; margin-bottom: 1rem;">
              {', '.join(category_keywords[cat])}
            </div>
            """
    return html

def generate_colored_summary(refined_label, school_vocab):
    vocab = school_vocab.get(refined_label, {})
    color_map = {
        "구성": "#e74c3c",    # 빨강
        "기법": "#3498db",    # 파랑
        "색감": "#f39c12",    # 노랑
        "주제": "#27ae60",    # 초록
    }

    html_output = ""
    for category in ["구성", "기법", "색감", "주제"]:
        keywords = vocab.get(category, [])
        if not keywords:
            continue
        keyword_list = ', '.join(keywords)
        color = color_map[category]
        html_output += f"""
        <div style='margin-bottom: 0.5rem;'>
            <strong style='color:{color}'>🟦 {category}</strong> – <span style='color:{color}'>{keyword_list}</span>
        </div>
        """
    return html_output


@app.post("/upload/", response_class=HTMLResponse)
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    # ✅ 하이브리드 분석 수행
    result = hybrid_art_style_analysis(file_path)

    # ✅ vocab 변수에 미리 저장
    school_vocab = structured_vocab.get(result['최종 학파'], {})
    artists = school_vocab.get("대표 예술가", [])
    formatted_text = format_gemini_text(result["Gemini 설명"])
    summary_html = generate_colored_summary(result["최종 학파"], structured_vocab)
    filtered_sentences = generate_summary_table(result["Gemini 설명"], school_vocab)

    result_html = f"""
    <html>
    <head><meta charset='utf-8'><title>AI 분석 결과</title></head>
    <body style="font-family: sans-serif; padding: 2rem; line-height: 1.6;">
      <h2>🎨 AI 분석 결과</h2>
      <img src="/uploads/{file.filename}" alt="업로드 이미지" style="max-width: 400px; border-radius: 8px; box-shadow: 0 0 8px #ccc;"/>
      <p><strong>최종 학파:</strong> {result['최종 학파']}</p>
      <p><strong>최종 유사도:</strong> {result['최종 유사도']}</p>
      <hr>
      <div style='padding: 1rem;'>
        <h3>📝 분석 설명 </h3>
        {summary_html}
      </div>
      <div style='margin-top: 2rem;'>
        <h3>📌 어휘사전 기반 주요 문장</h3>
        {filtered_sentences}
      </div>
      <hr>
      <h3>👨‍🎨 대표 예술가</h3>
      <p>{', '.join(artists)}</p>
    </body>
    </html>
    """
    return HTMLResponse(content=result_html)

@app.get("/")
def homepage():
    return HTMLResponse("""
    <html>
    <head>
      <meta charset="utf-8">
      <title>서양 미술사의 세계</title>
      <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    </head>
    <body class="bg-gray-100 text-gray-800 font-sans">
      <div class="min-h-screen flex items-center justify-center p-6">
        <div class="max-w-5xl w-full space-y-8">

          <div class="p-6 bg-blue-50 rounded-lg">
            <h2 class="text-2xl font-bold mb-4 text-blue-800">
              서양 미술사의 세계로 오신 것을 환영합니다
            </h2>
            <p class="mb-4">
              이 웹사이트는 중등 미술 교과과정에 맞춰 서양 미술사를 쉽고 재미있게 배울 수 있도록 구성되었습니다.
            </p>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div class="p-4 bg-white rounded-lg shadow-md">
                <h3 class="font-bold text-xl mb-2 text-blue-700">시대별 미술 탐구</h3>
                <p>르네상스부터 현대까지...</p>
                <a href="/periods" class="mt-4 inline-block px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                  이동
                </a>
              </div>
              <div class="p-4 bg-white rounded-lg shadow-md">
                <h3 class="font-bold text-xl mb-2 text-blue-700">작품 갤러리</h3>
                <p>유명 작품들을 감상하고...</p>
                <a href="/gallery" class="mt-4 inline-block px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                  이동
                </a>
              </div>
              <div class="p-4 bg-white rounded-lg shadow-md">
                <h3 class="font-bold text-xl mb-2 text-blue-700">AI 작품 분석</h3>
                <p>AI가 분석한 결과를...</p>
                <a href="/upload" class="mt-4 inline-block px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                  이동
                </a>
              </div>
            </div>
          </div>

        </div>
      </div>
    </body>
    </html>
    """)

@app.get("/periods")
def get_periods():
    vocab = structured_vocab  # 위에서 정의한 딕셔너리 사용

    html_cards = ""
    for period, details in vocab.items():
        card_html = f"""
        <div class="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow">
          <h3 class="text-xl font-bold mb-2 text-blue-800">{period}</h3>
        """

        for category in ['구성', '기법', '색감', '주제']:
            values = ', '.join(details.get(category, []))
            card_html += f"""
              <div class="mb-3">
                <h4 class="font-semibold">{category}</h4>
                <p class="text-sm">{values}</p>
              </div>
            """

        artists = details.get('대표 예술가', [])
        artist_tags = ""
        for idx, name in enumerate(artists[:3]):
            artist_tags += f'<span class="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">{name}</span> '

        if len(artists) > 3:
            extra_count = len(artists) - 3
            artist_tags += f'<span class="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">+{extra_count}명</span>'

        card_html += f"""
          <div>
            <h4 class="font-semibold">대표 예술가</h4>
            <div class="flex flex-wrap gap-1 mt-1">{artist_tags}</div>
          </div>
        </div>
        """

        html_cards += card_html

    return HTMLResponse(f"""
    <html>
    <head>
      <meta charset="utf-8">
      <title>서양 미술사의 주요 시대</title>
      <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    </head>
    <body class="bg-gray-100 p-8 font-sans">
      <div class="max-w-6xl mx-auto space-y-6">
        <h2 class="text-2xl font-bold mb-6">서양 미술사의 주요 시대</h2>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
          {html_cards}
        </div>
      </div>
    </body>
    </html>
    """)

@app.get("/gallery")
def gallery_page():
    # ① vocab 정의
    vocab = {
        '르네상스': {}, '바로크': {}, '로코코': {}, '신고전주의': {},
        '낭만주의': {}, '자연주의': {}, '사실주의': {}, '인상주의': {}, '입체파&추상화': {}
    }

    # ② artworks 정의
    artworks = [
        {
            "id": 1,
            "title": "모나리자",
            "artist": "레오나르도 다빈치",
            "period": "르네상스",
            "image": "/static/images/monalisa.jpg",
            "description": "원근법과 스푸마토 기법이 돋보이는 대표적인 르네상스 작품"
        },
        {
            "id": 2,
            "title": "야간 순찰",
            "artist": "렘브란트",
            "period": "바로크",
            "image": "/static/images/nightwatch.jpg",
            "description": "극적인 명암 대비와 역동적인 구도가 특징인 바로크 작품"
        },
        # 필요하면 더 추가
    ]

    # ③ 필터 버튼 생성
    html_buttons = ""
    for period in ['전체'] + list(vocab.keys()):
        html_buttons += f"""
        <button class="px-3 py-1 rounded-full text-sm bg-gray-200 text-gray-800 hover:bg-gray-300">
          {period}
        </button>
        """

    # ④ 카드 UI 생성
    html_cards = ""
    for artwork in artworks:
        html_cards += f"""
        <div class="bg-white rounded-lg shadow-md overflow-hidden">
          <div class="relative">
            <img src="{artwork['image']}" alt="{artwork['title']}" class="w-full h-64 object-cover" />
            <div class="absolute top-0 right-0 m-2 px-2 py-1 bg-blue-500 text-white text-sm rounded">
              {artwork['period']}
            </div>
          </div>
          <div class="p-4">
            <h3 class="text-xl font-bold mb-1">{artwork['title']}</h3>
            <p class="text-gray-600 mb-3">{artwork['artist']}</p>
            <p class="text-sm mb-4">{artwork['description']}</p>
            <button class="mt-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 w-full">
              자세히 보기
            </button>
          </div>
        </div>
        """

    # ⑤ 전체 HTML
    return HTMLResponse(f"""
    <html>
    <head>
      <meta charset="utf-8">
      <title>작품 갤러리</title>
      <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    </head>
    <body class="bg-gray-100 p-8 font-sans">
      <div class="max-w-6xl mx-auto space-y-6">
        <h2 class="text-2xl font-bold mb-6">작품 갤러리</h2>

        <div class="flex flex-wrap gap-2 mb-6">
          {html_buttons}
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
          {html_cards}
        </div>
      </div>
    </body>
    </html>
    """)

@app.get("/")
def read_root():
    return HTMLResponse("""
    <html>
    <head>
      <meta charset='utf-8'>
      <title>AI 미술 분석</title>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
      <script src="https://kit.fontawesome.com/a076d05399.js" crossorigin="anonymous"></script>
      <style>
        body {
          font-family: 'Inter', sans-serif;
          background: linear-gradient(to right, #eef2f3, #cfd9df);
          padding: 3rem;
          display: flex;
          justify-content: center;
          align-items: center;
          min-height: 100vh;
        }
        .container {
          background-color: #ffffff;
          border-radius: 16px;
          padding: 2.5rem 2rem;
          max-width: 600px;
          width: 100%;
          box-shadow: 0 8px 30px rgba(0,0,0,0.1);
          text-align: center;
        }
        .card {
          background: #fff;
          border-radius: 12px;
          padding: 1.5rem;
          box-shadow: 0 4px 20px rgba(0,0,0,0.05);
          margin-bottom: 1.5rem;
        }
        .card h3 {
          font-size: 1.25rem;
          color: #2c3e50;
        }
        .card p {
          color: #555;
          margin-top: 0.5rem;
          margin-bottom: 1rem;
        }
        .card a {
          background-color: #2980b9;
          color: white;
          padding: 0.5rem 1rem;
          border-radius: 8px;
          text-decoration: none;
        }
        .card a:hover {
          background-color: #1c5980;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h2><i class="fas fa-palette"></i> AI 기반 미술 사조 분석기</h2>
        <p>원하는 기능을 선택하세요.</p>

        <div class="card">
          <h3>AI 작품 분석</h3>
          <p>여러분이 직접 작품을 업로드하고 AI가 분석한 결과를 확인해보세요.</p>
          <a href="/upload">이동</a>
        </div>
      </div>
    </body>
    </html>
    """)

@app.get("/upload")
def upload_page():
    return HTMLResponse("""
    <html>
    <head>
      <meta charset='utf-8'>
      <title>AI 작품 업로드</title>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
      <script src="https://kit.fontawesome.com/a076d05399.js" crossorigin="anonymous"></script>
      <style>
        body {
          font-family: 'Inter', sans-serif;
          background: linear-gradient(to right, #eef2f3, #cfd9df);
          padding: 3rem;
          display: flex;
          justify-content: center;
          align-items: center;
          min-height: 100vh;
        }
        .container {
          background-color: #ffffff;
          border-radius: 16px;
          padding: 2.5rem 2rem;
          max-width: 600px;
          width: 100%;
          box-shadow: 0 8px 30px rgba(0,0,0,0.1);
          text-align: center;
        }
        input[type="file"] {
          margin-top: 1rem;
          padding: 0.7rem;
          border: 1px solid #ccc;
          border-radius: 8px;
          width: 100%;
        }
        input[type="submit"] {
          margin-top: 1.8rem;
          padding: 0.8rem 1.8rem;
          background-color: #34495e;
          border: none;
          color: white;
          border-radius: 10px;
          cursor: pointer;
          font-size: 1rem;
          transition: all 0.3s ease;
        }
        input[type="submit"]:hover {
          background-color: #2c3e50;
          transform: scale(1.03);
        }
        #preview {
          margin-top: 1.5rem;
          max-width: 100%;
          border-radius: 12px;
          box-shadow: 0 0 10px rgba(0,0,0,0.1);
          display: none;
        }
        #loading {
          margin-top: 1.2rem;
          color: #888;
          display: none;
        }
        #filename {
          margin-top: 0.8rem;
          font-size: 0.9rem;
          color: #555;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h2><i class="fas fa-upload"></i> 작품 업로드</h2>
        <p>작품 이미지를 업로드하면 AI가 분석해드려요.</p>
        <form action="/upload/" method="post" enctype="multipart/form-data" onsubmit="showLoading()">
          <input name="file" type="file" accept="image/*" onchange="previewImage(event)">
          <p id="filename"></p>
          <img id="preview" src="" alt="미리보기 이미지" />
          <input type="submit" value="분석 시작">
          <div id="loading">🔍 AI가 작품을 분석 중입니다...</div>
        </form>
      </div>

      <script>
        function previewImage(event) {
          const preview = document.getElementById('preview');
          const file = event.target.files[0];
          const filename = document.getElementById('filename');
          if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
              preview.src = e.target.result;
              preview.style.display = "block";
              filename.textContent = "📁 " + file.name;
            }
            reader.readAsDataURL(file);
          }
        }

        function showLoading() {
          document.getElementById('loading').style.display = 'block';
        }
      </script>
    </body>
    </html>
    """)

@app.post("/upload/")
def analyze_image(file: UploadFile = File(...)):
    filename = file.filename
    extension = filename.split(".")[-1]
    file_hash = sum([ord(c) for c in filename + extension]) % 10

    mock_periods = ["르네상스", "바로크", "로코코", "신고전주의", "낭만주의", "자연주의", "사실주의", "인상주의", "입체파&추상화"]
    predicted = mock_periods[file_hash % len(mock_periods)]

    return HTMLResponse(f"""
    <html>
    <head>
      <meta charset='utf-8'>
      <title>분석 결과</title>
      <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    </head>
    <body class="bg-gray-100 p-12">
      <div class="max-w-xl mx-auto bg-white p-6 rounded-lg shadow">
        <h2 class="text-2xl font-bold mb-4 text-blue-800">AI 분석 결과</h2>
        <p class="mb-2 text-gray-700">예측된 미술 사조: <span class="font-semibold">{predicted}</span></p>
        <p class="text-sm text-gray-500">(※ 이 결과는 예시이며 실제 AI 분석과는 다를 수 있습니다)</p>
        <a href="/" class="mt-6 inline-block text-blue-600 hover:underline">다시 분석하기</a>
      </div>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)