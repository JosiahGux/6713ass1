# z5643559 项目流程图

根据 `z5643559.py` 整理的六阶段流程图。

---

## 1. 程序总览（六阶段）

```mermaid
flowchart TB
    subgraph Phase1["阶段1: 模块导入"]
        A[main.py 导入 z5643559] --> B{__name__ != __main__?}
        B -->|是| C[打印 Initialising...]
        C --> Phase2
    end

    subgraph Phase2["阶段2: 加载数据与词表"]
        D[_get_level_vocab] --> E[load_training_data data.csv]
        E --> F[build_level_vocab]
        F --> G[level_vocab A1..C2]
    end

    subgraph Phase3["阶段3: 训练语言模型"]
        H[_get_lm] --> I[load_training_data]
        I --> J[_tokenize_text 每行]
        J --> K[BigramLM.train]
        K --> L[统计 bigram/unigram]
        L --> M[Laplace 平滑]
    end

    subgraph Phase4["阶段4: 单句转换入口"]
        N[transform_sentence] --> O[校验 source/target]
        O --> P[取 target_vocab]
        P --> Q[_get_nlp]
        Q --> R[分词 doc = nlp]
    end

    subgraph Phase5["阶段5: 逐词判断与替换"]
        R --> S[遍历每个 token]
        S --> T{是字母词 且 不在 target_vocab?}
        T -->|是| U[_best_replacement]
        T -->|否| V[保留原词]
        U --> W[_match_case]
        W --> X[拼入 rebuilt]
        V --> X
    end

    subgraph Phase6["阶段6: 输出"]
        X --> Y[last_word 更新]
        Y --> S
        S --> Z[return 拼接句]
    end

    Phase1 --> Phase2
    Phase2 --> Phase3
    Phase3 --> N
    Phase4 --> Phase5
    Phase5 --> Phase6
```

---

## 2. 阶段2：构建等级词表 (build_level_vocab)

```mermaid
flowchart LR
    A[data.csv] --> B[每行: cefr_level.strip]
    B --> C{level in CEFR_LEVELS?}
    C -->|是| D[_tokenize_text 取词]
    C -->|否| B
    D --> E[level_counts level 词频++]
    E --> F[每等级取前 15000 词]
    F --> G[level_vocab]
```

---

## 3. 阶段3：Bigram LM 训练 (BigramLM.train)

```mermaid
flowchart TB
    A[token_sequences] --> B[清空 bigram_counts, unigram_counts]
    B --> C[遍历每条 sequence]
    C --> D[插入 <s>]
    D --> E[逐词: bigram prev->w, unigram w]
    E --> F[插入 </s>]
    F --> G[计算 vocab_size]
    G --> H[完成]
```

---

## 4. 选替换词 (_best_replacement)

```mermaid
flowchart TB
    A[original_word, target_vocab, nlp, lm, prev_word] --> B{原词已在 target_vocab?}
    B -->|是| C[return None]
    B -->|否| D[nlp 取原词向量]
    D --> E{向量有效?}
    E -->|是| F[候选集 最多 8000]
    F --> G[对每个候选 w]
    G --> H[sim = cosine_similarity]
    H --> I[lm_score = lm.log_prob]
    I --> J[score = 0.6*sim + 0.4*norm_lm]
    J --> K[取 score 最大者]
    K --> L[return best_word]
    E -->|否| M[_best_replacement_wordnet]
    M --> N[WordNet 同义词 ∩ target_vocab]
    N --> L
```

---

## 5. transform_sentence 主流程

```mermaid
flowchart TB
    A[sentence, source_level, target_level] --> B[校验 CEFR]
    B --> C{source == target?}
    C -->|是| D[return sentence]
    C -->|否| E[_get_level_vocab]
    E --> F[target_vocab 或 回退并集]
    F --> G[_get_nlp]
    G --> H[_get_lm]
    H --> I[doc = nlp sentence]
    I --> J[last_word = None]
    J --> K[for tok in doc]
    K --> L{需替换?}
    L -->|是| M[_best_replacement]
    M --> N[_match_case]
    N --> O[rebuilt += text]
    L -->|否| O
    O --> P[更新 last_word]
    P --> K
    K --> Q[return join rebuilt]
```

---

## 6. 数据与依赖关系

```mermaid
flowchart LR
    subgraph 输入
        data[data.csv]
        sent[句子 + source/target]
    end

    subgraph 资源
        vocab[level_vocab]
        lm[BigramLM]
        nlp[SpaCy]
    end

    subgraph 输出
        out[转换后句子]
    end

    data --> vocab
    data --> lm
    sent --> nlp
    vocab --> out
    lm --> out
    nlp --> out
```

---

以上六图分别对应：**总览六阶段**、**等级词表构建**、**LM 训练**、**替换词选择**、**transform_sentence 主流程**、**数据与依赖**。可用支持 Mermaid 的编辑器（如 VS Code 插件）或 [mermaid.live](https://mermaid.live) 渲染查看。
