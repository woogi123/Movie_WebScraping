# streamlit run recommand_movies.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.graph_objects as go

# 초기 설정 
st.set_page_config(page_title="🍿 영화 추천 서비스 👀", layout="wide")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

#데이터 불러오기 
@st.cache_data
def load_data():
    df = pd.read_csv('movie_full.csv')
    df.columns = df.columns.str.strip().str.replace('\n', '')
    df.rename(columns={
        '감독연출': 'attraction_1',
        'OST': 'attraction_2',
        '배우연기': 'attraction_3',
        '영상미': 'attraction_4',
        '스토리': 'attraction_5',
        '스트레스 해소': 'emotion_1',
        '무서움': 'emotion_2',
        '몰입감': 'emotion_3',
        '긴장감': 'emotion_4',
        '현실감': 'emotion_5'
    }, inplace=True)
    df.fillna("No Data", inplace=True)    
    return df

df = load_data()


# 포인트 차트 함수 
def draw_radar_chart(labels, values):
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='red', linewidth=1)
    ax.fill(angles, values, color='pink', alpha=0.3)
    colors = plt.cm.jet(np.linspace(0, 1, len(values)-1))

    for i in range(len(values)-1):
        ax.scatter(angles[i], values[i], color=colors[i], s=5, edgecolor='black', zorder=5)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    return fig

# 성비 차트 함수
def draw_gender_chart(row):
    fig, ax = plt.subplots(figsize=(2, 2))
    labels_gender = ['남성', '여성']
    male_ratio_str = str(row['male_pct']).replace('남 ', '').replace('%', '').strip() if pd.notna(row['male_pct']) else '0'
    female_ratio_str = str(row['female_pct']).replace('여 ', '').replace('%', '').strip() if pd.notna(row['female_pct']) else '0'

    try:
        sizes_gender = [float(male_ratio_str), float(female_ratio_str)]
    except:
        sizes_gender = [0.0, 0.0]

    if sum(sizes_gender) == 0:
        ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center', fontsize=5)
        ax.axis('off')
    else:
        colors_gender = ['#66b3ff', '#ff9999']
        explode_gender = (0, 0.1)

        ax.pie( sizes_gender, explode=explode_gender, labels=labels_gender, colors=colors_gender, autopct='%1.1f%%', shadow=True, startangle=90)
        ax.axis('equal')
        plt.setp(ax.texts, fontsize=5)
    return fig

# 장르 중복 제거
def get_unique_genres(df):
    genres = df['genre'].dropna().astype(str).str.split(',')
    all_genres = [g.strip() for sublist in genres for g in sublist]
    return sorted(set(all_genres))
unique_genres = get_unique_genres(df)

def get_color(score):
    if score <= 20:
        return 'rgba(231, 76, 60, 0.8)'       # 붉은 계열
    elif score <= 30:
        return 'rgba(241, 196, 15, 0.8)'      # 노란 계열
    elif score <= 40:
        return 'rgba(243, 156, 18, 0.8)'      # 주황 계열
    else:
        return 'rgba(46, 204, 113, 0.8)'      # 연두 계열


# 사이드바 
st.sidebar.header("📝 Check your style")
selected_genre = st.sidebar.selectbox("🎯보고 싶은 장르 선택하기📌", unique_genres)

with st.sidebar.expander("✨ 매력 포인트 개인화"):
    attraction_values = [st.slider(label, 0, 50, 25) for label in ['감독연출', 'OST', '배우연기', '영상미', '스토리']]

with st.sidebar.expander("💖 감정 포인트 개인화"):
    emotion_values = [st.slider(label, 0, 50, 25) for label in ['스트레스 해소', '무서움', '몰입감', '긴장감', '현실감']]

search_button = st.sidebar.button("search")

# 탭 설정 
tab1, tab2, tab3, tab4 = st.tabs(["🎬 현재 상영 영화 TOP", "💯 맞춤 추천 결과", "🔍 영화 검색", "📈전체 평점 비교"])

# tab 1 
with tab1:
    if "page" not in st.session_state:
        st.session_state.page = 1
    page = st.session_state.page
    items_per_page = 6
    total_pages = (len(df) - 1) // items_per_page + 1
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(df))

    st.subheader("🎬 영화 목록")
    cols = st.columns(3)

    for i, idx in enumerate(range(start_idx, end_idx)):
        with cols[i % 3]:
            st.image(df['image_url'].iloc[idx], use_container_width=True)
            with st.expander(df['title'].iloc[idx]):
                st.markdown(f"""
                    **📜 예매율:** {df['percent'].iloc[idx]}  
                    **👍 감독:** {df['director'].iloc[idx]}  
                    **📍 출연진:** {df['actors'].iloc[idx]}  
                    **🎯 장르:** {df['genre'].iloc[idx]}  
                    **📆 개봉일:** {df['release_date'].iloc[idx]}  
                    **🎟️ 관람연령:** {df['rating'].iloc[idx]}  
                    **🧾 줄거리:** {df['story'].iloc[idx]}
                """)

    new_page = st.number_input("📃 페이지를 선택하세요", 1, total_pages, value=page, key="page_input")
    if new_page != page:
        st.session_state.page = new_page
        st.rerun()

# tab 2 
with tab2:
    if search_button:
        st.subheader("🎯 당신을 위한 영화 추천 결과")
        st.markdown("선택하신 매개변수와 유사도가 높은 콘텐츠를 분석했습니다.")

        feature_cols = ['attraction_1','attraction_2','attraction_3','attraction_4','attraction_5',
                        'emotion_1','emotion_2','emotion_3','emotion_4','emotion_5']

        df_filtered = df[df['genre'].apply(lambda x: selected_genre in str(x))]
        df_filtered[feature_cols] = df_filtered[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        user_vec = np.array(attraction_values + emotion_values).reshape(1, -1)
        sims = cosine_similarity(user_vec, df_filtered[feature_cols].to_numpy()).flatten()
        df_filtered['sim_score'] = sims

        top3 = df_filtered.nlargest(3, 'sim_score')

        cols = st.columns(3)
        for idx, (_, row) in enumerate(top3.iterrows()):
            with cols[idx]:
                st.image(row['image_url'], use_container_width=True)
                st.markdown(f"""
                    **{row['title']}**  
                    📜 예매율: {row['percent']}  
                    🔢 유사도 점수: {row['sim_score']:.3f}
                """)
                with st.expander("📖 자세히 보기"):
                    st.markdown(f"""
                        **🎯 장르:** {row['genre']}  
                        **😎 감독:** {row['director']}  
                        **👥 출연진:** {row['actors']}  
                        **📆 개봉일:** {row['release_date']}  
                        **🎟️ 관람연령:** {row['rating']}  
                        **🧾 줄거리:** {row['story']}  
                    """)
    else:
        st.info("👈 왼쪽에서 조건을 설정하고 'Search' 버튼을 눌러주세요!")

# tab 3 
with tab3:
    search_title = st.text_input("영화 제목", "ex. 타이타닉")

    if not search_title.strip():
        st.warning("😥 영화 제목을 입력해주세요.")
    else:
        matched = df[df['title'].str.contains(search_title, case=False, na=False)]
        if matched.empty:
            st.error("😢 해당 제목의 영화가 없습니다.")
        else:
            for _, row in matched.iterrows():
                with st.container():
                    img_col, info_col = st.columns([1.5, 2])
                    with img_col:
                        st.image(row['image_url'], use_container_width=True)
                    with info_col:
                        st.markdown(f"### 🎬 {row['title']}")
                        st.markdown(f"**감독:** {row['director']}")
                        st.markdown(f"**출연:** {row['actors']}")
                        st.markdown(f"**장르:** {row['genre']}")
                        st.markdown(f"**개봉일:** {row['release_date']}")
                        st.markdown(f"**관람연령:** {row['rating']}")
                        st.markdown(f"**예매율:** {row['percent']}")
                        st.markdown(f"**줄거리:** {row['story']}")

                        # 차트 데이터
                        charm_labels = ['스토리', '연출', '배우', 'OST', '영상미']
                        charm_cols = ['attraction_5', 'attraction_1', 'attraction_3', 'attraction_2', 'attraction_4']
                        charm_values = [float(row.get(col, 0)) if str(row.get(col, "")).replace('.', '', 1).isdigit() else 0 for col in charm_cols]

                        emotion_labels = ['스트레스 해소', '무서움', '몰입감', '긴장감', '현실감']
                        emotion_cols = ['emotion_1', 'emotion_2', 'emotion_3', 'emotion_4', 'emotion_5']
                        emotion_values = [float(row.get(col, 0)) if str(row.get(col, "")).replace('.', '', 1).isdigit() else 0 for col in emotion_cols]

                        def draw_total_score_chart(score):
                            color = get_color(score)

                            fig = go.Figure(go.Bar(
                                x=[score],
                                y=["평점"],
                                orientation='h',
                                marker=dict(
                                    color=color,
                                    line=dict(color='rgba(0,0,0,0.3)', width=1)
                                ),
                                text=f"{score:.1f}점",
                                textposition='auto'
                            ))

                            fig.update_layout(
                                xaxis=dict(range=[0, 100], title='점수'),
                                yaxis=dict(showticklabels=True),
                                height=100,
                                margin=dict(t=5, b=20, l=30, r=20)
                            )
                            return fig

                        # 매력 + 감정 통합
                        total_values = charm_values + emotion_values
                        average_score = round(np.nanmean(total_values), 2)  # 소수점 둘째 자리
                        
                        st.markdown("**종합 평점**")
                        st.plotly_chart(draw_total_score_chart(average_score), use_container_width=True)

                    charm_col, emotion_col, gender_col = st.columns(3)
                    with charm_col:
                        st.markdown("### ✨ 매력 포인트")
                        st.pyplot(draw_radar_chart(charm_labels, charm_values))
                    with emotion_col:
                        st.markdown("### 💖 감정 포인트")
                        st.pyplot(draw_radar_chart(emotion_labels, emotion_values))
                    with gender_col:
                        st.markdown("### 💁‍♀️/💁‍♂️ 성비 차트")
                        fig = draw_gender_chart(row)
                        st.pyplot(fig)

                    st.markdown("### 💬 관람객 리뷰")
                    reviews = str(row.get('reviews', '리뷰 없음'))
                    review_list = [rev.strip() for rev in reviews.split('/') if rev.strip()]
                    if review_list:
                        for rev in review_list:
                            st.write(f"👉 {rev}")
                    else:
                        st.write("아직 등록된 리뷰가 없습니다.")

# tab 4
with tab4:
    st.subheader("📊 전체 영화 평점 비교")

    # 종합 점수 계산
    cols = ['attraction_1','attraction_2','attraction_3','attraction_4','attraction_5',
            'emotion_1','emotion_2','emotion_3','emotion_4','emotion_5']

    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df['score'] = df[cols].fillna(0).sum(axis=1) / 10

    sorted_df = df.sort_values('score', ascending=False)
    colors = [get_color(score) for score in sorted_df['score']]

    fig = go.Figure(go.Bar(
        x=sorted_df['score'],
        y=sorted_df['title'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{s:.1f}점" for s in sorted_df['score']],
        textposition='auto'
    ))

    fig.update_layout(
        height=700,
        margin=dict(l=100, r=30, t=40, b=40),
        xaxis_title="점수",
        title="전체 영화 종합 점수 (색상별 구간 표시)",
        yaxis=dict(autorange="reversed")  # 높은 점수가 위로
    )

    st.plotly_chart(fig, use_container_width=True)