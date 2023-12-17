import streamlit as st
from config import FINETUNED_CHECKPOINT, BERT_CHECKPOINT, MAX_LEN, MAPPING
from utils import clean_text, preprocess, classify
from transformers import AutoModelForSequenceClassification, BertTokenizer



@st.cache(allow_output_mutation=True)
def load_model_tokenizer(model_checkpoint, tokenizer_checkpoint):
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_checkpoint)
    return model, tokenizer


model, tokenizer = load_model_tokenizer(FINETUNED_CHECKPOINT, BERT_CHECKPOINT)

language = st.selectbox('Select language :globe_with_meridians:',('Vietnamese','English'))
if language == "English":
    title = "Movie Reviews Classifier :movie_camera: :movie_camera: :movie_camera:"
    instruction = "Instructions: Enter a movie review :memo: and click the 'Process' button to classify it :rocket:."
    review = "🎬 Enter your movie review:"
    final = "Created by a group of K20 students in HCMUS in statistical machine learning"
    warning = 'Please enter a review before clicking Process.'
    button = "Process"
else:
    title = "Phân loại đánh giá phim :movie_camera: :movie_camera: :movie_camera:"
    instruction = "Hướng dẫn: Nhập đánh giá phim :memo: và nhấp vào nút 'Xử lý' để phân loại:rocket:."
    review = "🎬 Nhập đánh giá phim của bạn"
    final = "Được tạo ra bởi nhóm sinh viên K20 trường HCMUS trong môn Học Thống kê"
    warning = "Vui lòng nhập đánh giá trước khi nhấn Xử lý."
    button = "Xử lí"

st.title(title)

logo_url = "https://pito.vn/wp-content/uploads/2022/09/Logo-Dai-hoc-Khoa-hoc-Tu-nhien.webp"
st.image(logo_url, width=100)  

st.markdown(instruction)
review = st.text_area(review)

if st.button(button):
    if review.isspace() or len(review) == 0:
        st.warning(warning)
    else:
        if language == "English":
            st.markdown('## Review Classification Result')
            preprocessed_review = preprocess(review, tokenizer=tokenizer, max_len=MAX_LEN, clean_text=clean_text)
            out = classify(inputs=preprocessed_review, model=model, mapping=MAPPING)

            thumb = "⭐️" if out['Label'] == 'positive' else "👎"
            st.markdown(f"The review is **{out['Label']}** {thumb} with a confidence of **{out['Confidence']*100:.2f}%**.")
        else:
            st.markdown('# Kết quả phân loại')
            preprocessed_review = preprocess(review, tokenizer=tokenizer, max_len=MAX_LEN, clean_text=clean_text)
            out = classify(inputs=preprocessed_review, model=model, mapping=MAPPING)

            thumb = "⭐️" if out['Label'] == 'positive' else "👎"
            st.markdown(f"Bài đánh giá **{out['Label']}** {thumb} với độ chính xác là **{out['Confidence']*100:.2f}%**.")

st.markdown("---")
st.markdown(final)
