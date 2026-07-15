# Stock Price Prediction by OpenCode

LLMを活用した株価予測エージェント。企業のIR情報、株価推移、経済指標、国際情勢を総合的に分析し、1日後・1週間後・1か月後の株価を予測します。

## 特徴

- **IR情報スクレイピング**: 企業の公式IRページから決算資料・有価証券報告書を取得
- **経済指標の取得**: 為替(USD/JPY)、金利、原油価格、金価格、主要株価指数
- **国際情勢分析**: Reuters、BBC、日経のRSSから地政学リスクを自動検出
- **テクニカル分析**: 移動平均、モメンタム、ボラティリティの算出
- **数値予測モデル**: 複数コンポーネントを加重した方向スコアを算出

## 構成

```
src/stock_price_prediction/
├── main.py              # メインスクリプト（データ収集・出力）
├── data_sources.py      # Yahoo Finance API
├── forecast_model.py    # 数値予測モデル
├── company_master.py    # 企業ティッカーマスタ
├── ir_scraper.py        # IR情報スクレイピング
└── news_analyzer.py     # 国際情勢ニュース分析
```

## セットアップ

```bash
cd Stock_Price_Prediction_by_OpenCode
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -e .
```

## 使用法

### 基本的な使い方

```bash
python -m stock_price_prediction.main "トヨタ自動車"
```

### 対応企業

会社名で指定してください。以下の企業はローカルマスタに登録済みです：

| 企業名 | ティッカー |
|--------|------------|
| トヨタ自動車 | 7203.T |
| ソニーグループ | 6758.T |
| ソフトバンクグループ | 9984.T |
| 任天堂 | 7974.T |
| 日立製作所 | 6501.T |
| 三菱UFJ | 8306.T |
| キーエンス | 6861.T |
| 信越化学 | 4063.T |
| 三井住友 | 8316.T |
| Apple | AAPL |
| Microsoft | MSFT |
| NVIDIA | NVDA |
| Alphabet | GOOGL |
| Amazon | AMZN |
| Tesla | TSLA |

上記以外の企業もYahoo Finance検索で自動対応します。

### 出力形式

JSON形式で以下を出力します：

```json
{
  "ticker": "7203.T",
  "company_name": "Toyota Motor Corporation",
  "price_data": { ... },
  "financial_data": { ... },
  "macro_data": { ... },
  "ir_data": { ... },
  "geopolitical_data": { ... },
  "quant_forecast": { ... },
  "multi_horizon_forecast": { ... }
}
```

## LLMによる分析

このツールは**データ収集専用**です。LLM（GPT、Claudeなど）を直接使用しません。

### 分析の流れ

1. `python -m stock_price_prediction.main "企業名"` を実行
2. 出力されたJSONをクリップボードにコピー
3. LLM（ChatGPT、Claude、Geminiなど）に貼り付けて分析を依頼

### 分析依頼テンプレート

```
以下の株価データを分析して、1日後・1週間後・1か月後の株価予測を行ってください。

[JSONを貼り付け]

以下の観点から分析してください：
1. テクニカル分析（トレンド、移動平均、モメンタム）
2. ファンダメンタル分析（財務データ、業績予想）
3. マクロ経済（為替、金利、原油価格）
4. 国際情勢（地政学リスク）
5. IR情報（決算内容、業績予想）
```

## 予測モデルの仕組み

### コンポーネント加重

```
テクニカルモデル:     40%
財務・業績モデル:     25%
マクロモデル:         20%
ニュースイベント:     15%
```

### 予測期間ごとの重み

| 期間 | テクニカル | 財務 | マクロ | ニュース |
|------|------------|------|--------|----------|
| 明日 | 70% | 5% | 15% | 10% |
| 来週 | 55% | 10% | 20% | 15% |
| 来月 | 35% | 25% | 25% | 15% |
| 3ヶ月後 | 25% | 35% | 25% | 15% |

### 方向スコア

- **0.70以上**: 強い（Strong）
- **0.60-0.70**: やや強い（Slightly Strong）
- **0.60未満**: 中立（Neutral）

## 注意点

### 免責事項

- **投資助言ではありません**: このツールの出力は分析・研究用途の参考情報です
- **未校正モデル**: スコアは確率ではなく、モデル上の方向スコアです
- **将来の保証なし**: 過去のデータに基づく予測であり、将来の株価を保証するものではありません

### データの制限

- Yahoo Financeの無料APIを使用しているため、取得制限がある場合があります
- 一部の財務データが取得できない場合があります
- ニュースはRSSフィードの公開範囲のみです

### 注意すべき点

1. **短期的な予測は困難**: 株価は短期でランダムに変動します
2. **外部要因の影響**: 突発的なニュースやイベントで予測が外れる場合があります
3. **自己責任で判断**: 投資判断はご自身の責任で行ってください

## 開発者向け

### テスト実行

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

### 新規企業の追加

`src/stock_price_prediction/company_master.py` の `COMPANY_MASTER` リストに追加してください。

### IRページの追加

`src/stock_price_prediction/ir_scraper.py` の `KNOWN_IR_PAGES` ディクショナリに追加してください。

## ライセンス

MIT License
