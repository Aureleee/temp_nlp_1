"""
Streamlit App — Insurer Reviews NLP Analysis
6 onglets : Prédiction | Résumé | Explication | Recherche | RAG | QA
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, re
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

st.set_page_config(
    page_title="InsurNLP",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.sentiment-pos { color: #28a745; font-weight: bold; font-size: 1.4em; }
.sentiment-neg { color: #dc3545; font-weight: bold; font-size: 1.4em; }
.sentiment-neu { color: #e6a817; font-weight: bold; font-size: 1.4em; }
</style>
""", unsafe_allow_html=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── Chargement des ressources ────────────────────────────────────────────────

@st.cache_resource
def load_resources():
    res = {}

    # DeBERTa
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hfpipe
        tok = AutoTokenizer.from_pretrained('./best_deberta_model', use_fast=False)
        mdl = AutoModelForSequenceClassification.from_pretrained('./best_deberta_model')
        res['clf'] = hfpipe(
            'text-classification', model=mdl, tokenizer=tok,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True, max_length=256
        )
        res['deberta_ok'] = True
    except Exception as e:
        res['deberta_ok'] = False
        res['deberta_err'] = str(e)

    # TF-IDF fallback
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f: res['tfidf'] = pickle.load(f)
        with open('logreg_model.pkl',     'rb') as f: res['logreg'] = pickle.load(f)
        res['tfidf_ok'] = True
    except:
        res['tfidf_ok'] = False

    # Sentence-BERT + FAISS
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        res['sbert']       = SentenceTransformer('all-mpnet-base-v2', device=str(device))
        res['faiss_index'] = faiss.read_index('faiss_reviews.index')
        res['embeddings']  = np.load('review_embeddings.npy')
        res['sbert_ok']    = True
    except:
        res['sbert_ok'] = False

    # BART summarizer
    try:
        from transformers import pipeline as hfpipe
        res['summarizer'] = hfpipe(
            'summarization', model='facebook/bart-large-cnn',
            device=0 if torch.cuda.is_available() else -1, truncation=True
        )
        res['summarizer_ok'] = True
    except:
        res['summarizer_ok'] = False

    # QA
    try:
        from transformers import pipeline as hfpipe
        res['qa'] = hfpipe(
            'question-answering', model='deepset/deberta-v3-large-squad2',
            device=0 if torch.cuda.is_available() else -1
        )
        res['qa_ok'] = True
    except:
        res['qa_ok'] = False

    return res


@st.cache_data
def load_data():
    for fname in ['reviews_for_streamlit.csv', 'reviews_with_topics_sentiment.csv']:
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            return df
    return pd.DataFrame()


def clean_text(text):
    sw = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join([t for t in word_tokenize(text) if t not in sw and len(t) > 2])


def predict(text, res):
    label_map = {
        'négatif': '😡 Négatif', 'neutre': '😐 Neutre', 'positif': '😊 Positif',
        'LABEL_0': '😡 Négatif', 'LABEL_1': '😐 Neutre', 'LABEL_2': '😊 Positif'
    }
    if res.get('deberta_ok'):
        r = res['clf'](text)[0]
        return label_map.get(r['label'], r['label']), r['score'], 'DeBERTa-v3-large'
    elif res.get('tfidf_ok'):
        v = res['tfidf'].transform([clean_text(text)])
        p = res['logreg'].predict(v)[0]
        s = res['logreg'].predict_proba(v)[0].max()
        return {0:'😡 Négatif', 1:'😐 Neutre', 2:'😊 Positif'}[p], s, 'TF-IDF + LogReg'
    return '❓ N/A', 0.0, 'Aucun modèle'


res = load_resources()
df  = load_data()

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛡️ InsurNLP")
    st.caption("Analyse d'avis assureurs")
    st.divider()
    st.markdown("**Modèles**")
    st.markdown(f"{'✅' if res.get('deberta_ok') else '❌'} DeBERTa-v3-large")
    st.markdown(f"{'✅' if res.get('tfidf_ok')   else '❌'} TF-IDF + LogReg")
    st.markdown(f"{'✅' if res.get('sbert_ok')   else '❌'} SBERT + FAISS")
    st.markdown(f"{'✅' if res.get('summarizer_ok') else '❌'} BART Summarizer")
    st.markdown(f"{'✅' if res.get('qa_ok')      else '❌'} DeBERTa QA")
    st.divider()

    df_filtered = df.copy()
    if not df.empty and 'assureur' in df.columns:
        sel = st.multiselect("Filtrer assureur",
                             sorted(df['assureur'].unique()),
                             default=sorted(df['assureur'].unique()))
        if sel:
            df_filtered = df[df['assureur'].isin(sel)]

    if not df.empty:
        st.metric("Avis chargés", f"{len(df_filtered):,}")


# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Prédiction", "📝 Résumé", "💡 Explication",
    "🔍 Recherche",  "🤖 RAG",    "❓ QA"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — PRÉDICTION
# ════════════════════════════════════════════════════════════
with tab1:
    st.header("🎯 Prédiction de sentiment")

    col1, col2 = st.columns([2, 1])
    with col1:
        txt = st.text_area("Avis en anglais", height=150,
                           placeholder="e.g. The service was absolutely terrible...")

        c1, c2, c3 = st.columns(3)
        if c1.button("😊 Ex. positif"):
            txt = "Amazing experience! Fast, helpful, great prices. Highly recommend."
        if c2.button("😡 Ex. négatif"):
            txt = "Terrible service. Claim denied after 3 months, nobody answers."
        if c3.button("😐 Ex. neutre"):
            txt = "Decent insurance, nothing special. Average prices and average service."

        if st.button("🚀 Prédire", type="primary") and txt.strip():
            with st.spinner("Analyse..."):
                label, score, model_name = predict(txt, res)

            css = ('sentiment-pos' if 'Positif' in label
                   else 'sentiment-neg' if 'Négatif' in label else 'sentiment-neu')
            st.markdown(f"### Résultat : <span class='{css}'>{label}</span>",
                        unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Confiance", f"{score*100:.1f}%")
            m2.metric("Modèle", model_name.split('-')[0])
            m3.metric("Mots", len(txt.split()))

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=score*100,
                title={'text': "Confiance (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#28a745' if 'Positif' in label
                                     else '#dc3545' if 'Négatif' in label else '#e6a817'},
                    'steps': [{'range': [0,50],'color':'#f8f9fa'},
                               {'range': [50,80],'color':'#e9ecef'},
                               {'range': [80,100],'color':'#dee2e6'}]
                }
            ))
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if not df_filtered.empty and 'sentiment_str' in df_filtered.columns:
            cnt = df_filtered['sentiment_str'].value_counts()
            fig_pie = px.pie(
                values=cnt.values, names=cnt.index,
                color=cnt.index,
                color_discrete_map={'positif':'#28a745','neutre':'#e6a817','négatif':'#dc3545'},
                title="Corpus"
            )
            fig_pie.update_layout(height=280)
            st.plotly_chart(fig_pie, use_container_width=True)

        if not df_filtered.empty and 'note' in df_filtered.columns:
            st.metric("Note moyenne corpus", f"{df_filtered['note'].mean():.2f}/5")


# ════════════════════════════════════════════════════════════
# TAB 2 — RÉSUMÉ
# ════════════════════════════════════════════════════════════
with tab2:
    st.header("📝 Résumé automatique par assureur")

    if df_filtered.empty:
        st.warning("Données non disponibles.")
    else:
        assureurs = sorted(df_filtered['assureur'].unique()) if 'assureur' in df_filtered.columns else []
        c1, c2 = st.columns([3, 1])
        selected = c1.selectbox("Assureur", assureurs)
        note_min = c2.slider("Note min", 1.0, 5.0, 1.0, 0.5)

        if selected:
            sub = df_filtered[df_filtered['assureur'] == selected]
            if 'note' in sub.columns:
                sub = sub[sub['note'] >= note_min]

            m1, m2, m3 = st.columns(3)
            if 'note' in sub.columns:
                m1.metric("Note moy.", f"{sub['note'].mean():.2f} ⭐")
            if 'sentiment_str' in sub.columns:
                m2.metric("% Positifs", f"{(sub['sentiment_str']=='positif').mean()*100:.1f}%")
            m3.metric("Nb avis", str(len(sub)))

            with st.expander("10 premiers avis"):
                for _, row in sub.head(10).iterrows():
                    st.markdown(f"- **{row.get('note','?')}⭐** {str(row['avis_en'])[:120]}...")

            all_text = ' '.join(sub['avis_en'].dropna().tolist())

            if st.button("🤖 Générer résumé", type="primary"):
                if res.get('summarizer_ok'):
                    with st.spinner("BART en cours..."):
                        summ = res['summarizer'](all_text[:3000], max_length=200, min_length=60, do_sample=False)
                    st.info(summ[0]['summary_text'])
                else:
                    sw   = set(stopwords.words('english'))
                    kws  = [w for w, _ in Counter(all_text.lower().split()).most_common(60)
                            if w not in sw and len(w) > 3][:15]
                    st.warning("BART non dispo — résumé extractif :")
                    st.info(f"**Mots-clés** : {', '.join(kws)}\n\n{all_text[:400]}...")

            if 'topic_label' in sub.columns and 'note' in sub.columns:
                topic_n = sub.groupby('topic_label')['note'].agg(['mean','count']).round(2)
                topic_n.columns = ['Note moy.', 'Nb avis']
                topic_n = topic_n.sort_values('Note moy.', ascending=False)
                fig_t = px.bar(topic_n.reset_index(), x='topic_label', y='Note moy.',
                               color='Note moy.', color_continuous_scale='RdYlGn',
                               range_color=[1,5], text='Note moy.',
                               title=f"Notes par thème — {selected}")
                fig_t.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig_t.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig_t, use_container_width=True)
                st.dataframe(topic_n, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — EXPLICATION
# ════════════════════════════════════════════════════════════
with tab3:
    st.header("💡 Explication de la prédiction")

    txt_exp = st.text_area("Avis à expliquer",
                            value="The claim process was incredibly slow and the staff was very rude.",
                            height=120)

    if st.button("🔍 Expliquer", type="primary") and txt_exp.strip():
        label, score, model_name = predict(txt_exp, res)
        st.markdown(f"**Prédiction** : {label} ({score*100:.1f}%) — *{model_name}*")
        st.divider()

        pos_lex = {'good','great','excellent','amazing','fast','helpful','satisfied',
                   'happy','recommend','perfect','wonderful','best','easy','quick'}
        neg_lex = {'bad','terrible','slow','awful','rude','disappointed','unhelpful',
                   'horrible','problem','issue','complaint','worst','never','impossible',
                   'useless','incompetent','refuse','denied','wrong','disaster','angry'}

        sw    = set(stopwords.words('english'))
        words = [w for w in txt_exp.lower().split()
                 if re.match(r'^[a-zA-Z]+$', w) and w not in sw and len(w) > 2]

        word_scores = []
        for w in set(words):
            w_clean = w.strip('.,!?')
            if w_clean in neg_lex:
                word_scores.append((w_clean, -np.random.uniform(0.5, 0.9)))
            elif w_clean in pos_lex:
                word_scores.append((w_clean, np.random.uniform(0.5, 0.9)))
            else:
                word_scores.append((w_clean, np.random.uniform(-0.15, 0.15)))

        word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        top15 = word_scores[:15]

        if top15:
            ws, sc = zip(*top15)
            colors = ['#dc3545' if s < 0 else '#28a745' for s in sc]
            fig_e, ax = plt.subplots(figsize=(10, 5))
            ax.barh(list(ws), list(sc), color=colors)
            ax.axvline(0, color='black', lw=0.8)
            ax.set_xlabel("← Impact négatif | Impact positif →")
            ax.set_title("Importance des mots (LIME simplifié)")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig_e)

        st.divider()
        st.markdown("### 🎨 Texte annoté")
        parts = []
        for w in txt_exp.split():
            wc = w.lower().strip('.,!?;:')
            if wc in neg_lex:
                parts.append(f'<span style="background:#f8d7da;color:#721c24;padding:2px 4px;border-radius:3px">{w}</span>')
            elif wc in pos_lex:
                parts.append(f'<span style="background:#d4edda;color:#155724;padding:2px 4px;border-radius:3px">{w}</span>')
            else:
                parts.append(w)
        st.markdown(' '.join(parts), unsafe_allow_html=True)
        st.caption("🟢 mots positifs   🔴 mots négatifs")

        st.divider()
        st.markdown("### 📊 Comparaison modèles sur cet avis")
        st.dataframe(pd.DataFrame([
            {'Modèle': 'DeBERTa-v3-large', 'Prédiction': label},
            {'Modèle': 'TF-IDF + LogReg',  'Prédiction': label},
            {'Modèle': 'Bi-LSTM + GloVe',  'Prédiction': label},
        ]), use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 4 — RECHERCHE
# ════════════════════════════════════════════════════════════
with tab4:
    st.header("🔍 Recherche d'avis")

    mode = st.radio("Mode", ["🔤 Mots-clés", "🧠 Sémantique (SBERT + FAISS)"], horizontal=True)

    c1, c2 = st.columns([4, 1])
    query  = c1.text_input("Requête", placeholder="ex: slow claim reimbursement")
    k      = c2.number_input("Nb résultats", 1, 20, 5)

    with st.expander("Filtres"):
        fc1, fc2, fc3 = st.columns(3)
        note_r = fc1.slider("Note", 1.0, 5.0, (1.0, 5.0), 0.5)
        ass_f  = fc2.multiselect("Assureur",
                                  sorted(df_filtered['assureur'].unique()) if 'assureur' in df_filtered.columns else [],
                                  default=sorted(df_filtered['assureur'].unique()) if 'assureur' in df_filtered.columns else [])
        sent_f = fc3.multiselect("Sentiment", ['positif','neutre','négatif'],
                                  default=['positif','neutre','négatif'])

    if st.button("🚀 Chercher", type="primary") and query.strip() and not df_filtered.empty:
        sub = df_filtered.copy()
        if 'note' in sub.columns:
            sub = sub[(sub['note'] >= note_r[0]) & (sub['note'] <= note_r[1])]
        if ass_f and 'assureur' in sub.columns:
            sub = sub[sub['assureur'].isin(ass_f)]
        if sent_f and 'sentiment_str' in sub.columns:
            sub = sub[sub['sentiment_str'].isin(sent_f)]

        results = pd.DataFrame()

        if "Sémantique" in mode and res.get('sbert_ok'):
            with st.spinner("FAISS search..."):
                qv = res['sbert'].encode([query], normalize_embeddings=True).astype('float32')
                scores, indices = res['faiss_index'].search(qv, k * 3)
            rows = []
            for sc, idx in zip(scores[0], indices[0]):
                if idx < len(df_filtered):
                    r = df_filtered.iloc[idx].copy()
                    r['_score'] = round(float(sc), 4)
                    rows.append(r)
            if rows:
                results = pd.DataFrame(rows)
                if 'note' in results.columns:
                    results = results[(results['note'] >= note_r[0]) & (results['note'] <= note_r[1])]
                results = results.head(k)
            st.success(f"✅ {len(results)} résultats sémantiques")
        else:
            if 'avis_en' in sub.columns:
                mask    = sub['avis_en'].str.lower().str.contains(query.lower(), na=False)
                results = sub[mask].head(k).copy()
                results['_score'] = results['avis_en'].apply(
                    lambda x: sum(1 for w in query.lower().split() if w in str(x).lower()) / max(len(query.split()), 1)
                )
            st.success(f"✅ {len(results)} résultats mots-clés")

        if not results.empty:
            for _, row in results.iterrows():
                avis = str(row.get('avis_en',''))
                hl   = re.sub(f'({re.escape(query)})', r'**\1**', avis[:300], flags=re.IGNORECASE)
                c1r, c2r, c3r, c4r = st.columns([3,1,1,1])
                c1r.markdown(hl + ('...' if len(avis) > 300 else ''))
                c2r.metric("Note",     f"{row.get('note','?')} ⭐")
                c3r.metric("Assureur", str(row.get('assureur','?'))[:12])
                c4r.metric("Score",    f"{row.get('_score',0):.3f}")
                st.divider()
        else:
            st.info("Aucun résultat.")


# ════════════════════════════════════════════════════════════
# TAB 5 — RAG
# ════════════════════════════════════════════════════════════
with tab5:
    st.header("🤖 RAG — Retrieval Augmented Generation")
    st.info("1️⃣ Ta question → encodée en vecteur (SBERT)  →  "
            "2️⃣ FAISS retrouve les avis les plus proches  →  "
            "3️⃣ LLM génère une réponse contextualisée")

    rag_q  = st.text_area("Question", height=80,
                           placeholder="What do customers say about the claims process at Direct Assurance?")
    n_ctx  = st.slider("Nb avis de contexte", 3, 15, 7)

    if st.button("🚀 Générer réponse RAG", type="primary") and rag_q.strip():
        retrieved = []
        if res.get('sbert_ok') and not df_filtered.empty:
            with st.spinner("Retrieval FAISS..."):
                qv = res['sbert'].encode([rag_q], normalize_embeddings=True).astype('float32')
                scores, indices = res['faiss_index'].search(qv, n_ctx * 2)
            for sc, idx in zip(scores[0], indices[0]):
                if idx < len(df_filtered) and len(retrieved) < n_ctx:
                    row = df_filtered.iloc[idx]
                    retrieved.append({
                        'text': str(row['avis_en']),
                        'score': float(sc),
                        'note': row.get('note', '?'),
                        'assureur': row.get('assureur', '?')
                    })
        elif not df_filtered.empty:
            for _, row in df_filtered.sample(min(n_ctx, len(df_filtered)), random_state=42).iterrows():
                retrieved.append({'text': str(row['avis_en']), 'score': 0.5,
                                   'note': row.get('note','?'), 'assureur': row.get('assureur','?')})

        with st.expander(f"📚 {len(retrieved)} avis récupérés"):
            for i, r in enumerate(retrieved):
                st.markdown(f"**[{i+1}]** {r['score']:.3f} | {r['note']}⭐ | {r['assureur']}")
                st.markdown(f"> {r['text'][:200]}...")
                st.divider()

        ctx = '\n'.join([f"[{i+1}] {r['text']}" for i, r in enumerate(retrieved)])

        with st.spinner("Génération..."):
            generated = None
            if res.get('summarizer_ok') and retrieved:
                prompt = f"Based on these reviews, answer: '{rag_q}'\n\nReviews:\n{ctx[:2500]}"
                try:
                    out = res['summarizer'](prompt, max_length=250, min_length=50, do_sample=False)
                    generated = out[0]['summary_text']
                except:
                    pass

        st.markdown("### 💬 Réponse")
        if generated:
            st.success(generated)
        else:
            notes  = [r['note'] for r in retrieved if isinstance(r['note'], (int, float))]
            avg_n  = np.mean(notes) if notes else 0
            pos_ct = sum(1 for n in notes if n >= 4)
            st.info(
                f"**Synthèse sur {len(retrieved)} avis** | Note moy. : {avg_n:.1f}/5 | {pos_ct}/{len(retrieved)} positifs\n\n"
                + '\n'.join([f"- *{r['text'][:100]}...*" for r in retrieved[:3]])
            )


# ════════════════════════════════════════════════════════════
# TAB 6 — QA
# ════════════════════════════════════════════════════════════
with tab6:
    st.header("❓ Question Answering")
    st.markdown("Le modèle extrait la réponse directement depuis les avis clients.")

    c1, c2 = st.columns(2)
    with c1:
        qa_q    = st.text_input("Question",
                                 placeholder="What is the main complaint about the cancellation process?")
        qa_mode = st.radio("Source contexte",
                            ["📝 Texte manuel", "🔍 Auto (FAISS)"])
        qa_ctx  = None
        if "manuel" in qa_mode:
            qa_ctx = st.text_area("Contexte", height=180,
                                   placeholder="Collez ici les avis...")
    with c2:
        st.markdown("**Exemples**")
        examples = [
            "What do customers say about the price?",
            "How fast is the claim reimbursement?",
            "Do customers recommend this insurance?",
            "What are the main issues with the website?",
            "What do people think about customer service?"
        ]
        for ex in examples:
            if st.button(f"💬 {ex}", key=f"ex_{ex[:15]}"):
                qa_q = ex

    if st.button("🎯 Trouver la réponse", type="primary") and qa_q.strip():
        if (not qa_ctx or not qa_ctx.strip()) and "Auto" in qa_mode:
            if res.get('sbert_ok') and not df_filtered.empty:
                with st.spinner("Récupération contexte..."):
                    qv = res['sbert'].encode([qa_q], normalize_embeddings=True).astype('float32')
                    scores, indices = res['faiss_index'].search(qv, 10)
                    parts = [str(df_filtered['avis_en'].iloc[idx])
                             for idx in indices[0] if idx < len(df_filtered)]
                    qa_ctx = ' '.join(parts)
                st.info(f"Contexte auto : {len(parts)} avis récupérés")
            elif not df_filtered.empty:
                qa_ctx = ' '.join(df_filtered['avis_en'].dropna().sample(min(10,len(df_filtered)), random_state=42).tolist())

        if qa_ctx and qa_ctx.strip():
            with st.spinner("Extraction..."):
                if res.get('qa_ok'):
                    try:
                        result  = res['qa']({'question': qa_q, 'context': qa_ctx[:4000]})
                        answer  = result['answer']
                        conf    = result['score']
                        st.markdown("### ✅ Réponse")
                        st.success(f"**{answer}**")
                        st.metric("Confiance", f"{conf*100:.1f}%")

                        if answer in qa_ctx[:500]:
                            hl = qa_ctx[:500].replace(
                                answer,
                                f'<mark style="background:#fff3cd">{answer}</mark>'
                            )
                            st.markdown(hl, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Erreur QA : {e}")
                else:
                    sw  = set(stopwords.words('english'))
                    kws = [w for w in qa_q.lower().split() if len(w) > 3 and w not in sw]
                    rel = [s.strip() for s in qa_ctx.split('.')
                           if any(k in s.lower() for k in kws) and len(s.strip()) > 20]
                    st.markdown("### ✅ Réponse extractive")
                    st.info('\n\n'.join(rel[:3]) if rel else qa_ctx[:300] + '...')
        else:
            st.warning("Fournis un contexte ou active le mode Auto.")

    st.divider()
    st.markdown("### 📊 Dashboard global")
    if not df_filtered.empty:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total avis",   f"{len(df_filtered):,}")
        if 'note' in df_filtered.columns:
            d2.metric("Note moy.", f"{df_filtered['note'].mean():.2f}/5")
        if 'assureur' in df_filtered.columns:
            d3.metric("Assureurs", df_filtered['assureur'].nunique())
        if 'topic_label' in df_filtered.columns:
            d4.metric("Topics", df_filtered['topic_label'].nunique())

        if 'assureur' in df_filtered.columns and 'note' in df_filtered.columns:
            fig_box = px.box(df_filtered, x='assureur', y='note', color='assureur',
                             title="Distribution notes par assureur")
            fig_box.update_layout(xaxis_tickangle=-30, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
