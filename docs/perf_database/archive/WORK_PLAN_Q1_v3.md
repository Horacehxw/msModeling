# 绠楀瓙鎬ц兘鏁版嵁搴?Q1 宸ヤ綔璁″垝 (v3)

**鐩爣**: 2026.3.23 瀹屾垚绔埌绔泦鎴愶紝DeepSeek-V3 / Qwen3-32B 浠跨湡璇樊 <15%
**鍩哄噯鏃ユ湡**: 2026.3.5锛堝懆鍥涙櫄鍙戝竷锛?.6 璧锋墽琛岋級
**鍥㈤槦**: 6 浜猴紙1 SE + 5 寮€鍙戯級
**鍛ㄦ湡**: 3.6-3.23锛堜笁涓?Phase锛?
**璁捐鏂囨。**: `OPERATOR_PERF_DATABASE_DESIGN_zh_v1.2.md`锛堝悓鐩綍锛?
**绌垮埡鎬荤粨**: `reports/spike_executive_summary_zh.md`

---

## 鐩綍

- [1. 椤圭洰姒傝](#1-椤圭洰姒傝)
- [2. 杩涘睍绠＄悊](#2-杩涘睍绠＄悊)
- [3. 鍥㈤槦涓庤亴璐(#3-鍥㈤槦涓庤亴璐?
- [4. 浠诲姟渚濊禆鎬昏](#4-浠诲姟渚濊禆鎬昏)
- [5. Phase 1锛氭牳蹇冮泦鎴?+ Mini 楠岃瘉锛?.6-3.13锛塢(#5-phase-1鏍稿績闆嗘垚--mini-楠岃瘉36-313)
- [5.5 C10 鍚庣画璁″垝锛欻CCL 鏁版嵁鍏ュ簱涓庨獙璇乚(#55-c10-鍚庣画璁″垝hccl-鏁版嵁鍏ュ簱涓庨獙璇?
- [6. Phase 2锛氭暟鎹墿鍏?+ 铻嶅悎 Pass + DSV3 娣卞害鍖归厤锛?.16-3.20锛塢(#6-phase-2鏁版嵁鎵╁厖--铻嶅悎-pass--dsv3-娣卞害鍖归厤316-320)
- [7. Phase 3锛氱鍒扮绮惧害楠岃瘉锛?.19-3.23锛塢(#7-phase-3绔埌绔簿搴﹂獙璇?19-323)
- [8. 椋庨櫓涓庣紦瑙(#8-椋庨櫓涓庣紦瑙?
- [闄勫綍 A锛氬垎鏀瓥鐣(#闄勫綍-a鍒嗘敮绛栫暐)
- [闄勫綍 B锛氬弬鑰冪储寮昡(#闄勫綍-b鍙傝€冪储寮?
- [闄勫綍 C锛氳繘灞曠鐞嗙粏鍒橾(#闄勫綍-c杩涘睍绠＄悊缁嗗垯)

---

## 1. 椤圭洰姒傝

### 1.1 鐩爣涓庣幇鐘?

涓?TensorCast 鏋勫缓鍩轰簬瀹炴祴 Profiling 鏁版嵁鐨勭畻瀛愭€ц兘浼扮畻绯荤粺锛坄EmpiricalPerformanceModel + DataSource` 妯″紡锛岃璁℃枃妗?搂1.1锛夈€俤b-spike 绌垮埡宸查獙璇佹灦鏋勫彲琛屾€э細Qwen3-32B BF16 Prefill 鍖归厤鐜?87%锛岃绠楃畻瀛?100%銆傛牳蹇冧唬鐮?production-ready锛岀洿鎺ヤ綔涓轰骇鍝佸垎鏀熀纭€銆?

### 1.2 鏍稿績鍋囪

| 鍋囪 | 楠岃瘉鏂瑰紡 | 鑻ヤ笉鎴愮珛鐨勫奖鍝?|
|------|---------|-------------|
| CommAnalytic 鍦?Qwen3 Prefill 涓婄簿搴﹀彲鎺ュ彈 | Phase 1 mini 绔埌绔獙璇?| 閫氫俊鏌ヨ璺緞浼樺厛绾ч渶鎻愬墠 |
| DSV3 W8A8 op_mapping 鍙閲忓畬鎴?| C3/C4 鏄犲皠楠岃瘉 | 鏄犲皠宸ヤ綔閲忕炕鍊?|
| MC2 鍦?compile pass 涓凡姝ｇ‘铻嶅悎 | XJT楠岃瘉 | 闇€璋冩暣 composite fallback |
| 閫氱敤 shape 绾挎€ф彃鍊?+ FIA sqrt 鍙樻崲鍙弧瓒虫墍鏈夊満鏅紙涓嶉渶瑕?per-operator 缁村害澹版槑锛?| TCX鎻掑€肩簿搴︽祴璇?+ ZZY/HDY override 鏍囨敞 | 闇€鏂板 kernel_overrides |

### 1.3 浜や粯鏍囧噯

| 鎸囨爣 | 鐩爣鍊?|
|-----|-------|
| 绔埌绔€楁椂璇樊 | <15%锛堝姣斿疄闄?vLLM Profiling锛?|
| 鍗曠畻瀛愯宸紙宸插尮閰嶏級 | <20% |
| 鏃堕棿瑕嗙洊鐜?| >90% |

### 1.4 鏈€缁堜氦浠樼墿锛?.23锛?

| 浜や粯鐗?| 楠屾敹鏍囧噯 |
|-------|---------|
| CLI `--performance-model profiling --compile` | Qwen3-32B + DSV3 绔埌绔彲杩愯 |
| 绮惧害鎶ュ憡锛圦wen3-32B + DSV3锛?| 绔埌绔宸?<15% |
| 瀹屾暣鏁版嵁搴擄紙CSV + YAML锛?| 瑕嗙洊 Tier 1/2 绠楀瓙锛堣璁℃枃妗?搂7.1锛?|
| database validation tool | 鍙噸澶嶉獙璇?|
| 鏁版嵁閲囬泦宸ュ叿閾撅紙7 涓伐鍏凤級 | 鍙噸澶嶆墽琛?|

---

## 2. 杩涘睍绠＄悊

- **椋炰功鏃ユ姤**锛氭瘡浜烘瘡澶╂洿鏂拌繘灞?闃诲/椋庨櫓淇″彿锛堣瑙乕闄勫綍 C](#闄勫綍-c杩涘睍绠＄悊缁嗗垯)锛?
- **绔欎細**锛氫粎璁ㄨ闃诲椤瑰拰椋庨櫓锛孭hase 1/3 姣忔棩锛孭hase 2 闅旀棩
- **DIMA 鐪嬫澘**锛氫换鍔″崱鐗囩姸鎬佸悓姝ワ紝瀵?MY 鍚堜綔鏂瑰彲瑙?
- **Review 鑺傜偣**锛?.13 Phase 1 Review 鈫?3.19 Phase 2 Review 鈫?3.23 浜や粯 Review

---

## 3. 鍥㈤槦涓庤亴璐?

### 3.1 鍒嗗伐鎬昏〃

| 浜哄憳 | 鎶曞叆 | 鑱岃矗鍩?| 浠ｇ爜 Owner |
|------|------|--------|-----------|
| **ZH** | 50% | DataSource 鏌ヨ寮曟搸锛歚_lookup_compute` / `_lookup_comm` / `_lookup_composite` + review 鍏ㄩ儴鏌ヨ浠ｇ爜 PR | `perf_database/*.py` |
| **TCX** | 100% | 鏁版嵁灞傚叏閾捐矾锛氬伐鍏烽摼 + Microbenchmark + Attention 鏌ヨ涓庢暟鎹?+ 鍩虹鎻掑€硷紱鍗忓姪 SE 杩涘睍绠＄悊锛堟棩鎶ヨ窡韪€佺珯浼氳褰曪級 | `tools/perf_data_collection/`, attention 鏌ヨ, 鎻掑€?|
| **ZZY** | 100% | Qwen3 op_mapping锛欱F16 鍦烘櫙楠岃瘉 + Decode 鎵╁睍 + 鑷姩鍖栨柟妗?spec | `op_mapping.yaml` (Qwen3), 楠岃瘉鎶ュ憡 |
| **HDY** | 100% | DSV3 op_mapping + HCCL锛歐8A8 鏄犲皠 + 閫氫俊鏁版嵁閲囬泦 + DSV3 Profiling 鍒嗘瀽 | `op_mapping.yaml` (DSV3), HCCL 鏁版嵁 |
| **XJT** | 70% | 闆嗘垚灞傦細CLI + compile pass 铻嶅悎锛圡C2 楠岃瘉, KvRmsNormRopeCache锛?| CLI, `compilation/` |
| **HXW** | SE | spec review + 鍐崇瓥 + 杩涘睍绠＄悊锛堜笉 own 浜у搧浠ｇ爜锛?| 鈥?|

### 3.2 鍗忎綔鍏崇郴涓庢帴鍙?

```
XJT(闆嗘垚灞?  ZH(鏌ヨ灞?  TCX(鏁版嵁灞?  ZZY(Qwen3鏄犲皠)  HDY(DSV3鏄犲皠)
 CLI/Pass        lookup寮曟搸    CSV宸ュ叿/鎻掑€?    op_mapping楠岃瘉      op_mapping+HCCL
    |                |         Attn鏌ヨ              |                    |
    |                |              |                |                    |
    +--- pass 浜у嚭 --+-- 鏌ヨ鍚堝叆 --+-- mapping 鍚屾 -+--------------------+
```

**鎺ュ彛鐐?*锛堥渶 PR review 鍗忚皟鐨勫湴鏂癸級锛?
- TCX 鈫?ZH锛歚_lookup_attention()` 浠ｇ爜鍚堝叆 `profiling_data_source.py`
- TCX 鈫?ZH锛欼nterpolatingDataSource 浠ｇ爜鍚堝叆 `perf_database/`
- ZZY/HDY 鈫?ZH锛歚op_mapping.yaml` 鍙樻洿褰卞搷鏌ヨ閫昏緫鏃堕渶鍚屾
- XJT 鈫?ZH锛氭柊澧?compile pass 浜х敓鐨?TC op 闇€鍚屾鍒?`op_mapping.yaml`

HXW锛圫E锛夛細鍐崇瓥 + 杩涘睍绠＄悊锛圱CX鍗忓姪锛夛紱涓?own 浜у搧浠ｇ爜锛屾寜闇€鍙備笌鎶€鏈璁恒€?

### 3.3 鎶€鏈柟妗堢‘璁?

姣忎釜鎶€鏈柟妗堢敱璐熻矗浜鸿嚜琛岃捣鑽夊苟楠岃瘉銆傞獙璇佹柟寮忥細瀵圭収璁捐鏂囨。瀵瑰簲绔犺妭 + 绌垮埡鎶ュ憡宸叉湁缁撹锛屽湪鏃ユ姤涓畝瑕佽鏄庢柟妗堣鐐瑰拰楠岃瘉缁撴灉鍗冲彲銆傛湁鐤戦棶鎴栧垎姝ф椂鍦ㄧ珯浼氭彁鍑鸿璁恒€?

| 鏂规 | 璐熻矗浜?| 楠岃瘉渚濇嵁 | 瀹屾垚鏃堕棿 |
|------|--------|---------|---------|
| 17 椤圭畝鍖栬瘎浼?| HXW | 绌垮埡鎶ュ憡 搂5 | 3.6 |
| 璁＄畻+閫氫俊铻嶅悎绠楀瓙纭 | HDY | DSV3 Profiling CSV 涓悳绱㈣绠?閫氫俊铻嶅悎绫?kernel Type锛堝惈 MC2 鍙婂叾浠栬瀺鍚堝舰寮忥級 | 3.6锛?h锛?|
| Attention 鍖归厤瑙勫垯 | TCX | 绌垮埡鎶ュ憡 搂4.1 + 璁捐鏂囨。 搂4.8锛屽啓鍗曞厓娴嬭瘯楠岃瘉 | 3.9 |
| 閫氫俊鏁版嵁琛ㄦ牸寮?| ZH | 璁捐鏂囨。 搂4.4 + 搂4.7锛屽鐓?`comm_config_example.yaml` | 3.9 |
| MoE/MLA 鍖归厤瑙勫垯 | ZH | 璁捐鏂囨。 搂4.2 composite 鍒嗚В琛紝鍐欏崟鍏冩祴璇曢獙璇?| 3.12 |

---

## 4. 浠诲姟渚濊禆鎬昏

### 4.1 渚濊禆鍥?

```
          db-spike 宸叉湁浠ｇ爜 (feat/perf-database 鍩虹)
                    |
    +---------------+---------------+---------------+
    v               v               v               v
 A1 CLI          B1 閫氫俊鏌ヨ     C1+C2 绠楀瓙娓呭崟   D1 瑙ｆ瀽楠岃瘉
 (XJT)        (ZH)          (寮?鑳?骞惰)     (TCX)
    |               |               |               |
    v               v               v               v
 A2 绔埌绔?      B2 Composite    C3 Qwen3楠岃瘉    D2 Attention
 (XJT)        (ZH)          (ZZY)         鏌ヨ瀹炵幇
    |               |          C7 DSV3鏄犲皠       (TCX)
    v               |          (HDY)             |
 A3 铻嶅悎merge       |               |               v
 + MC2楠岃瘉          v               v            D3 鍩虹鎻掑€?
 (XJT)       B1+B2 瀹屾垚     C3+C7+C8瀹屾垚     (TCX)
    |               |               |               |
    +-------+-------+-------+-------+-------+-------+
            v                                       v
   Phase 1 浜や粯 + Mini 绔埌绔獙璇?(3.13)
            |
    +-------+-------+-------+-------+
    v       v       v       v       v
  E1-E4   F1      G1-G2   H1-H4   E5
  鏁版嵁    铻嶅悎    MoE/MLA  鍒嗘瀽    Attn鎻掑€?
  (TCX)(XJT)(ZH) (寮?鑳?  (TCX)
    |       |       |       |       |
    +-------+-------+-------+-------+
            v
   Phase 2 浜や粯 + DSV3 Mini 楠岃瘉 (3.19)
            |
    +-------+-------+
    v       v       v
  J1 Qwen3 J2 DSV3 J3 淇
  (绁?璁?  (鑳?璁?  (寮犲垎鏋?绁?鍞愪慨澶?
            v
   J4 绮惧害鎶ュ憡 (3.23)
```

### 4.2 鍏抽敭璺緞

`A1 鈫?A2 鈫?A3 鈫?Mini 楠岃瘉 鈫?G1 鈫?J2 鈫?J4`

### 4.3 Phase 鏃堕棿绾?

```
3.5(鍙戝竷)  3.6 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ 3.13        3.16 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ 3.19  3.20 鈹€鈹€鈹€鈹€ 3.23
           鈫愨攢鈹€ Phase 1 鈹€鈹€鈫?Review    鈫愨攢鈹€ Phase 2 鈹€鈹€鈫?Review       浜や粯
                                                 鈫愨攢鈹€ Phase 3 鈹€鈹€鈹€鈹€鈫?
```

**Phase 1 浠诲姟鎺掑竷锛?.6-3.13锛?*锛?

```
      3.6       3.9       3.10      3.11      3.12      3.13
       |         |         |         |         |         |
XJT |-- A1 ---|------ A2 ---------|-- A3+MC2楠岃瘉 ------|
       |         |         |         |         |         |
ZH   |         |--- B1 ------------|--- B2 ------------|
       |         |         |         |         |         |
TCX |-- D1 ---|-- D2 Attention ---|-- D3 鎻掑€?--|D4---|
       |         |         |         |         |         |
ZZY |         |= C1+C2 =|--- C3 Qwen3楠岃瘉 ---|C4+C5--|
       |         |         |         |         |         |
HDY |C6+MC2鏌?|--- C9 --|--- C7 DSV3鏄犲皠 ----|C8+C10-|
```

---

## 5. Phase 1锛氭牳蹇冮泦鎴?+ Mini 楠岃瘉锛?.6-3.13锛? 涓伐浣滄棩锛?

**鐩爣**锛欳LI 绔埌绔彲杩愯 + 璁＄畻/閫氫俊/Attention/Composite 鍥涙潯鏌ヨ璺緞 + op_mapping 鍙屾ā鍨嬮獙璇?+ 鍩虹鎻掑€?+ **Mini 绔埌绔娆¤窇閫?*銆?

---

### 浠诲姟 A锛欳LI 闆嗘垚 + Compile Pass锛圶JT锛?0%锛?

**鐩爣**锛氳 `--performance-model profiling --compile` 绔埌绔彲杩愯锛屽苟楠岃瘉宸叉湁铻嶅悎 pass 姝ｇ‘宸ヤ綔銆?

**鑳屾櫙**锛歞b-spike 宸叉湁 `empirical.py`锛?7 琛岋級鍜?CLI 鏀瑰姩鍙傝€冦€俙--compile` 鏄纭娇鐢?profiling 妯″紡鐨勫墠鎻愶紙绌垮埡鎶ュ憡 搂3.3锛夈€侻C2 pass 宸叉湁瀹屾暣瀹炵幇锛坄compilation/freezing_passes/patterns/matmul_allreduce.py`锛?61 琛岋紝5 绉嶉噺鍖栧彉浣擄級锛岄渶楠岃瘉鍏朵笌 profiling 鏁版嵁鐨勫榻愩€?

**淇敼鑼冨洿**锛歚tensor_cast/scripts/text_generate.py`, `tensor_cast/core/model_runner.py`, `tensor_cast/core/config_resolver.py`

**鍙傝€?*锛氳璁℃枃妗?搂5.1-搂5.3锛圕LI 鎺ュ彛锛夈€伮?.1锛堣瀺鍚?Gap 鐘舵€侊級

| # | 妫€鏌ョ偣 | 瀹屾垚鏃ユ湡 | 楠屾敹鏍囧噯 |
|---|-------|---------|---------|
| A1 | CLI `--performance-model {analytic,profiling}` + `--perf-database` 璺緞鍙傛暟 | 3.9 | analytic 琛屼负涓嶅彉锛沺rofiling 妯″紡鍒涘缓 EmpiricalPerformanceModel |
| A2 | 绔埌绔細Qwen3-32B Prefill `--performance-model profiling --compile` | 3.11 | 涓嶆姤閿欙紝log_stats 杈撳嚭鍛戒腑鐜?|
| A3 | 铻嶅悎 Pass merge锛歋wiGlu + GroupedMatmul+SwiGlu 浠?develop 鍚堝叆 + MC2 pass 楠岃瘉 | 3.13 | 鍗曞厓娴嬭瘯閫氳繃锛沗tensor_cast.matmul_all_reduce` 鍑虹幇鍦?dispatch trace 涓?|

**璁＄畻+閫氫俊铻嶅悎绠楀瓙楠岃瘉瑕佺偣**锛?
- 纭 `--compile` 鍚?dispatch trace 涓嚭鐜?`tensor_cast.matmul_all_reduce`锛堜笉鍐嶆槸鍒嗙鐨?mm + all_reduce锛?
- HDY 3.6 纭 DSV3 Profiling 涓槸鍚︽湁璁＄畻+閫氫俊铻嶅悎绫?kernel Type锛堝惈 MC2 鍙婂叾浠栬瀺鍚堝舰寮忥級
- **缁撹锛堝凡纭锛?*锛?
  - **MC2锛圡atMul+AllReduce 铻嶅悎锛?*锛欴SV3 Profiling 涓棤涓撶敤 kernel Type锛宮atmul锛坄QuantBatchMatmulV3`锛夊拰閫氫俊锛坄hcom_reduceScatter_` / `hcom_allGather_`锛夊垎寮€璁板綍 鈫?淇濈暀 `composite: true` + `sub_kernels: [QuantBatchMatmulV3, hcom_allReduce_]` 鍒嗚В鏌ヨ
  - **DispatchFFNCombine锛堣绠?閫氫俊铻嶅悎锛?*锛欴SV3 Profiling 涓?*瀛樺湪**姝よ瀺鍚?kernel锛岃瀺鍚堜簡 `all_to_all脳2 + GroupedMatmul脳2 + SwiGlu + MoE routing`锛岃€楁椂鍗犵鍒扮 **35.3%**锛屾槸 DSV3 鏈€閲嶈鐨勫崟涓€ kernel銆俆C 灏嗗叾鍒嗚В涓?`permute_tokens + grouped_matmul脳2 + swiglu + unpermute_tokens + all_to_all脳2`锛宱p_mapping.yaml 宸查厤缃?`composite: true` 澶勭悊锛屾棤闇€鏂板鐩存帴鏄犲皠

---

### 浠诲姟 B锛欴ataSource 鏌ヨ璺緞锛圸H锛?0%锛?

**鐩爣**锛氬湪宸叉湁 `_lookup_compute()` 鍩虹涓婏紝鏂板閫氫俊鏌ヨ鍜?Composite 鏌ヨ涓ゆ潯璺緞銆?

**鑳屾櫙**锛氬綋鍓?`profiling_data_source.py` 鐨?`lookup()` 涓紝`communication` 鍜?`composite` 涓や釜鍒嗘敮鐩存帴 return None锛堢┛鍒虹畝鍖栭」 S-11/S-12锛夈€?

**淇敼鑼冨洿**锛歚tensor_cast/performance_model/perf_database/profiling_data_source.py`

**鍙傝€?*锛氳璁℃枃妗?搂4.2锛堟煡璇㈠垎娲撅級銆伮?.4锛堥€氫俊鏌ヨ锛夈€伮?.7锛堥€氫俊 CSV 鏍煎紡锛?

**鍓嶇疆渚濊禆**锛氶€氫俊鏁版嵁琛?spec锛圸H鑷繁璧疯崏锛?.9 鍓嶅畬鎴愶紝HXW review锛?

| # | 妫€鏌ョ偣 | 瀹屾垚鏃ユ湡 | 楠屾敹鏍囧噯 |
|---|-------|---------|---------|
| B1 | `_lookup_comm()`锛氫粠 OpInvokeInfo 璁＄畻 message_bytes + topology_tier锛屾煡璇㈤€氫俊 CSV | 3.11 | 鍗曞厓娴嬭瘯锛歛ll_reduce/all_gather 杩斿洖鑰楁椂 |
| B2 | `_lookup_composite()`锛歮atmul_all_reduce 鍒嗚В + MLA 鍒嗚В妗嗘灦 | 3.13 | 鍗曞厓娴嬭瘯锛歮atmul_all_reduce 鍒嗚В鍚庡尮閰?|

**閫氫俊鏌ヨ瀹炵幇瑕佺偣**锛堣璁℃枃妗?搂4.2锛夛細
- `args[0]` 鈫?`message_bytes = tensor.nelement() * tensor.element_size()`
- `rank_group` 浣嶇疆鍥犵畻瀛愯€屽紓锛歛ll_reduce=args[2], all_gather=args[3], all_to_all=args[4]
- `topology_tier = comm_grid._get_topology_idx_for_group(rank_group)`
- CSV 鎸?`(num_devices, topology_tier)` 绮剧‘鍖归厤

**Composite 鏌ヨ瀹炵幇瑕佺偣**锛堣璁℃枃妗?搂4.2锛夛細
- `composite: true` 鏃跺垎瑙ｄ负 sub_kernels 閫愪釜鏌ヨ骞舵眰鍜?
- MLA 鍒嗚В澶嶇敤 `performance_model/__init__.py` 宸叉湁 shape 鎺ㄥ閫昏緫
- 浠讳竴瀛愬唴鏍告湭鍛戒腑 鈫?鏁翠綋 return None 鈫?fallback analytic

**璇存槑**锛歚_lookup_attention()` 鐢盩CX瀹炵幇锛堜换鍔?D2锛夛紝鎻愪氦 PR 鍚嶼H review 骞跺悎鍏ャ€?

---

### 浠诲姟 C锛歰p_mapping 绯荤粺鍖栭獙璇侊紙ZZY + HDY锛屽悇 100%锛?

**鐩爣**锛氱郴缁熸€ч獙璇佸凡鏈?op_mapping 鏄犲皠锛岃ˉ鍏?DSV3 Decode W8A8 鍦烘櫙鏄犲皠銆傝繖鏄鍒扮绮惧害鐨?*鏍稿績鐡堕** 鈥?鏄犲皠閿欒鐩存帴瀵艰嚧绠楀瓙 MISS銆?

**鑳屾櫙**锛氱┛鍒洪樁娈靛缓绔嬩簡 60+ 鏉℃槧灏勶紝浣嗕粎鍦?Qwen3 BF16 Prefill 涓婇獙璇併€侱SV3 Decode 鏈?10+ 涓柊 kernel type 闇€瑕佹槧灏勶紙QuantBatchMatmulV3, GroupedMatmul, DequantSwigluQuant 绛夛級銆?

**鍙傝€?*锛?
- **鏄犲皠鏂规硶璁?*锛歚tutorial/OP_PLUGIN_MAPPING_TUTORIAL.md`锛堟鍚?鍙嶅悜鏄犲皠鎿嶄綔鎵嬪唽 + 閫熸煡琛級
- **鏄犲皠鏍煎紡**锛氳璁℃枃妗?搂4.5锛坥p_mapping.yaml 瑙勬牸锛?
- **鏄犲皠绀轰緥**锛歚examples/op_mapping_example.yaml`
- **绠楀瓙鍒嗙骇**锛氳璁℃枃妗?搂7.1-搂7.2锛圱ier 1/2/3 + 鍗犳瘮鏁版嵁锛?

**淇敼鑼冨洿**锛歚perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/v0.13.0/op_mapping.yaml`

**楠岃瘉鏂规硶璁?*锛堟瘡鏉℃槧灏勭殑楠岃瘉姝ラ锛夛細
1. 浠?Profiling 鎻愬彇 kernel Type 鍙婂叾 Input Shapes / Data Types
2. 鐢?analytic 妯″紡璺?TC锛屽鍑?dispatch trace锛屾壘鍒板搴旂殑 TC op 鍙婂叾 args shapes
3. 鎸?`tutorial/OP_PLUGIN_MAPPING_TUTORIAL.md` 搂6-7 纭 TC op 鈫?kernel Type 鐨勬槧灏勯摼
4. 瀵规瘮 TC args shapes 涓?Profiling Input Shapes锛岃褰曞樊寮傦紙batch 缁村害銆丗RACTAL_NZ銆乸adding 绛夛級
5. 纭宸紓鍙 `profiling_data_source.py` 鐨勯€氱敤瑙勫垯澶勭悊锛堢┛鍒烘姤鍛?搂3.1 鍏被宸紓锛?

#### ZZY锛圦wen3 涓荤嚎锛?

| # | 妫€鏌ョ偣 | 瀹屾垚鏃ユ湡 | 楠屾敹鏍囧噯 |
|---|-------|---------|---------|
| C1 | Qwen3 Profiling 绠楀瓙娓呭崟锛歍op-20 (Type, 璋冪敤娆℃暟, 鑰楁椂鍗犳瘮) | 3.9 | 琛ㄦ牸杈撳嚭 |
| C2 | TC dispatch trace 瀵煎嚭锛歛nalytic 妯″紡璺?Qwen3-32B Prefill + Decode | 3.9锛圕1/C2 骞惰锛?| trace 鏃ュ織 |
| C3 | BF16 鍦烘櫙閫愭潯鏄犲皠楠岃瘉锛氭寜楠岃瘉鏂规硶璁洪€愭潯妫€鏌?| 3.11 | 楠岃瘉鎶ュ憡锛堝凡楠岃瘉/闇€娉ㄦ剰/涓嶅尮閰嶏級 |
| C4 | Qwen3 Decode 鍦烘櫙鏄犲皠琛ュ厖 + 楠岃瘉 | 3.12 | op_mapping 瑕嗙洊 Qwen3 Decode Top-15 |
| C5 | op_mapping 鑷姩鍖栨柟妗?spec + 浼樺寲 OP_PLUGIN_MAPPING_TUTORIAL | 3.13 | spec 鏂囨。 + 鏁欑▼澧炶ˉ DSV3 瀹炰緥 |

#### HDY锛圖SV3 涓荤嚎 + HCCL锛?

| # | 妫€鏌ョ偣 | 瀹屾垚鏃ユ湡 | 楠屾敹鏍囧噯 |
|---|-------|---------|---------|
| C6 | DSV3 Profiling 绠楀瓙娓呭崟锛歍op-20 鎺掑簭琛?| 3.6锛堝揩閫熶换鍔★級 | 琛ㄦ牸杈撳嚭 |
| 璁＄畻+閫氫俊铻嶅悎纭 | 鏌?DSV3 Profiling 鏄惁鏈夎绠?閫氫俊铻嶅悎绫?kernel Type锛堝惈 MC2 鍙婂叾浠栬瀺鍚堝舰寮忥級 | 3.6锛?h锛?| 缁撹 鈫?鍛婄煡XJT鍜孼H |
| C9 | HCCL 鏁版嵁閲囬泦鏂规锛歚generate_comm_microbench.py` 瀹炵幇 | 3.10 | 鑴氭湰鍙繍琛?|
| C7 | DSV3 W8A8 op_mapping 鎵╁睍锛歈uantBatchMatmulV3, AscendQuantV2, DequantSwigluQuant, GroupedMatmul, TransposeBatchMatMul, MoeGatingTopK 绛?| 3.12 | op_mapping 瑕嗙洊 DSV3 Top-15 |
| C8 | W8A8 閲忓寲鍦烘櫙鏄犲皠楠岃瘉 | 3.13 | 楠岃瘉鎶ュ憡 |
| C10 | HCCL 闆嗙兢鏁版嵁閲囬泦锛? 绉嶉€氫俊绠楀瓙 x 鍚?topology_tier锛?| 3.13 | CSV 浜у嚭锛堣 搂C10 鍚庣画璁″垝锛?|

**鎻掑€?override 鏍囨敞**锛氬垎鏋?op_mapping 鏃堕『渚跨‘璁ゅ悇 kernel_type 鏄惁闇€瑕佹彃鍊肩壒娈婂鐞嗭紙`interpolation_policy.kernel_overrides`锛夈€傞鏈熺粨鏋滐細浠?FusedInferAttentionScore 闇€瑕?sqrt 鍙樻崲锛屽叾浣欑畻瀛愬潎閫傜敤榛樿绾挎€ф彃鍊笺€?

**鍙屼汉浜ゅ弶楠岃瘉**锛歓ZY review HDY鐨?DSV3 鏄犲皠锛孒DY review ZZY鐨?Qwen3 鏄犲皠銆?

---

### 浠诲姟 D锛氭暟鎹噰闆嗗伐鍏烽摼 + Attention + 鎻掑€硷紙TCX锛?00%锛?

**鐩爣**锛氶獙璇佹暟鎹В鏋愬伐鍏?+ 瀹炵幇 Attention 鏌ヨ + 瀹炵幇鍩虹鎻掑€?+ 绠楀瓙鍙戠幇宸ュ叿銆?

**鑳屾櫙**锛氬綋鍓?7 涓伐鍏蜂腑鍙湁 `parse_kernel_details.py` 瀹屾暣锛?42 琛岋級锛屽叾浠?6 涓槸 stub銆侫ttention (`FusedInferAttentionScore`) 鏄?Prefill 涓欢杩熸渶楂樼殑鍗曠畻瀛愶紙~100us+锛夛紝瀵圭鍒扮绮惧害褰卞搷鏈€澶э紙绌垮埡鎶ュ憡 搂4.1锛夈€傛彃鍊兼槸瀹炵敤鎬х殑鍏抽敭鐡堕锛堢┛鍒烘姤鍛?S-17锛夈€?

**淇敼鑼冨洿**锛?
- `tools/perf_data_collection/parse_kernel_details.py`, `operator coverage check tool`
- `tensor_cast/performance_model/perf_database/profiling_data_source.py`锛坄_lookup_attention()` 鏂规硶锛?
- `tensor_cast/performance_model/perf_database/interpolating_data_source.py`

**鍙傝€?*锛?
- Attention锛氳璁℃枃妗?搂4.8锛團usedAttention 鐗规畩澶勭悊锛夈€佺┛鍒烘姤鍛?搂4.1
- 鎻掑€硷細璁捐鏂囨。 搂4.4锛圛nterpolatingDataSource锛夈€丄I Configurator 瀹炵幇锛坄src/aiconfigurator/sdk/perf_database.py` 鎻掑€兼柟娉曪級

| # | 妫€鏌ョ偣 | 瀹屾垚鏃ユ湡 | 楠屾敹鏍囧噯 |
|---|-------|---------|---------|
| D1 | `parse_kernel_details.py` 楠岃瘉锛氬湪 Qwen3 + DSV3 鏁版嵁涓婄‘璁よ緭鍑烘纭?| 3.6 | 杈撳嚭 CSV 涓?db-spike 宸叉湁鏁版嵁涓€鑷?|
| D2 | `_lookup_attention()` 瀹炵幇 | 3.10 | 鍗曞厓娴嬭瘯锛歈wen3 Prefill FIA 鍛戒腑 |
| D3 | InterpolatingDataSource 鍩虹鐗堬細鏈€杩戦偦 + 绾挎€ф彃鍊?| 3.12 | 鍗曞厓娴嬭瘯锛歴eq=200 杩斿洖浼扮畻鍊?|
| D4 | `operator coverage check tool`锛氬姣?Profiling Type vs op_mapping.yaml | 3.13 | known 绠楀瓙瑕嗙洊 >90% 璋冪敤娆℃暟 |

**D2 Attention 鏌ヨ瀹炵幇瑕佺偣**锛堣璁℃枃妗?搂4.8锛夛細
- 浠?`OpInvokeInfo.args[6]`锛坰eq_lens锛夎绠?`batch_size = len(seq_lens)` 鍜?`avg_seq_len = mean(seq_lens)`
- 浠?`OpInvokeInfo.args[0]`锛坬uery tensor锛夋彁鍙?`num_heads`, `head_dim`
- FIA CSV 绱㈠紩缁村害锛歚(batch_size, avg_seq_len, num_heads, head_dim, dtype)`
- 鍖哄垎 PA锛圥agedAttention, decode, seq_lens 闀匡級鍜?FA锛團lashAttention, prefill, query_lens 闀匡級
- 鎻愪氦 PR 鍚庣敱ZH review 骞跺悎鍏?`profiling_data_source.py`

**D3 鎻掑€煎疄鐜拌鐐?*锛堝弬鑰?AI Configurator + 璁捐鏂囨。 搂4.4锛夛細
- Wrapper 妯″紡鍖呰 ProfilingDataSource锛氱簿纭懡涓?鈫?鐩存帴杩斿洖锛屾湭鍛戒腑 鈫?鎻掑€?
- **閫氱敤鎻掑€奸€昏緫锛堜笉闇€瑕?per-operator 缁村害澹版槑锛?*锛歞type+format 绮剧‘鍖归厤锛堝凡鍦?ProfilingDataSource 瀹炵幇锛夛紝shape 缁村害鍋氭渶杩戦偦鎼滅储 + 绾挎€ф彃鍊?
- 璇诲彇 `op_mapping.yaml` 鐨?`interpolation_policy.kernel_overrides` 搴旂敤鐗规畩鍙樻崲锛堝綋鍓嶄粎 FIA 闇€瑕?sqrt锛?
- 鎻愪氦 PR 鍚庣敱ZH review 骞跺悎鍏?`perf_database/`

---

### Phase 1 閲岀▼纰戯紙3.13锛?

**蹇呰揪浜や粯鐗?*锛?

| 浜や粯鐗?| 楠屾敹鏍囧噯 | 璐熻矗浜?|
|-------|---------|--------|
| CLI `--performance-model profiling` | 绔埌绔彲杩愯 | XJT |
| `_lookup_comm()` | 鍗曞厓娴嬭瘯閫氳繃 | ZH |
| `_lookup_composite()` | matmul_all_reduce 鍒嗚В閫氳繃 | ZH |
| `_lookup_attention()` | Qwen3 Prefill FIA 鍛戒腑 | TCX 鈫?ZH review |
| InterpolatingDataSource 鍩虹鐗?| 绾挎€ф彃鍊煎彲鐢?| TCX 鈫?ZH review |
| op_mapping 楠岃瘉鎶ュ憡锛圦wen3 BF16锛?| 瑕嗙洊 Top-15 | ZZY |
| op_mapping 鎵╁睍锛圖SV3 W8A8锛?| 瑕嗙洊 Top-15 | HDY |
| HCCL 鏁版嵁 | 闆嗙兢閲囬泦瀹屾垚 | HDY |
| 铻嶅悎 Pass merge + MC2 楠岃瘉 | 鍗曞厓娴嬭瘯閫氳繃 | XJT |

**Mini 绔埌绔獙璇侊紙3.13锛屽叏鍛橈級**锛?

鐢ㄥ凡鏈夋暟鎹窇 Qwen3-32B Prefill 绔埌绔紝璁板綍锛?

| 鎸囨爣 | 璁板綍鍐呭 |
|------|---------|
| 鍛戒腑鐜?| HIT / MISS / FALLBACK 鍚勫灏?|
| Fallback 绠楀瓙鑰楁椂鍗犳瘮 | 鍝簺绠楀瓙璧颁簡 analytic fallback锛屽崰绔埌绔櫨鍒嗘瘮 |
| 宸插尮閰嶇畻瀛愯宸?| 涓?Profiling 瀹炴祴瀵规瘮 |
| 绔埌绔垵濮嬭宸?| 鍏佽杩滆秴 15%锛岄噸鐐规毚闇茬郴缁熸€ч棶棰?|

**Go/No-Go**锛氳嫢 >50% 绠楀瓙 MISS 鎴?fallback 鍗犳瘮 >30%锛孭hase 2 浼樺厛绾ч渶閲嶆帓銆?

---

## 5.5 C10 鍚庣画璁″垝锛欻CCL 鏁版嵁鍏ュ簱涓庨獙璇?

> **鑳屾櫙**锛欳10 鍒濇閲囬泦锛?026.3.11锛夊凡浜у嚭 4 涓€氫俊绠楀瓙 CSV锛坅ll_reduce / all_gather / reduce_scatter / all_to_all锛夛紝瑕嗙洊 tier=1锛坕ntra_pod锛?6 鍗★級銆傛暟鎹垎鏋愬彂鐜拌嫢骞茶川閲忛棶棰橈紝闇€鍦?H3 浜ゅ弶楠岃瘉鍓嶅畬鎴愪慨澶嶅拰琛ラ噰銆?

### 鏁版嵁璐ㄩ噺鐜扮姸

| 鏂囦欢 | 琛屾暟 | 闂 |
|------|------|------|
| `hcom_allReduce_.csv` | 22锛堥噸澶嶏級 | 涓ゆ torchrun append锛岄渶鍘婚噸锛?MB/256MB/512MB 鏈夊紓甯稿€?|
| `hcom_allGather_.csv` | 11 | 4KB/16KB 楂樺欢杩燂紙HCCL JIT 鍒濆鍖栵級锛?MB/4MB 鍋忔參 |
| `hcom_reduceScatter_.csv` | 10 | 缂?512MB锛?KB/16MB 寮傚父 |
| `hcom_allToAll_.csv` | 11 | 鏂囦欢鍚嶉敊璇紙搴斾负 `hcom_alltoallv_.csv`锛夛紱4KB/16KB 楂樺欢杩?|

**鏍规湰鍘熷洜**锛氭棫鑴氭湰姣忎釜 op 鐙珛 torchrun锛孒CCL 姣忔閲嶆柊鍒濆鍖栵紝灏忔秷鎭懡涓?JIT 缂栬瘧寮€閿€銆?

### 鑴氭湰淇锛堝凡瀹屾垚锛宑ommit f16a6ac锛?

| 淇椤?| 璇存槑 |
|--------|------|
| 鍗?session 杩愯 | 鎵€鏈?op + message_sizes 鍚堝苟涓轰竴娆?torchrun锛孒CCL 鍙垵濮嬪寲涓€娆?|
| 鍏ㄥ眬棰勭儹 | 姣忎釜 (op, group) 鍏堣窇涓€娆?1KB 瑙﹀彂 HCCL JIT 缂栬瘧锛屽啀寮€濮嬫寮忚鏃?|
| WARMUP_ITERS 10鈫?0 | 姣忎釜 message_size 鐨勯鐑疆娆″姞鍊?|
| tier=2 瑕嗙洊 | 鏂板 `--num-devices 2`锛岄噰闆?die_level锛堝悓 node 鍐?2 鍗★級鏁版嵁 |
| 鏂囦欢鍚嶄慨姝?| `_OP_TO_CSV_FILENAME` 鏄犲皠 `all_to_all 鈫?hcom_alltoallv_.csv` |

### 鍚庣画浠诲姟娓呭崟

| # | 浠诲姟 | 璐熻矗浜?| 鎴 | 楠屾敹鏍囧噯 |
|---|------|--------|------|---------|
| C10-1 | 閲嶆柊閲囬泦锛歚bash run_comm_bench.sh ./hccl_data_v2`锛堝崟 session锛屽惈 tier=2锛?| HDY | 3.14 | 4 涓?CSV锛屾瘡涓?22 琛岋紙11 sizes 脳 2 tiers锛夛紝鏃犻噸澶嶈 |
| C10-2 | 鏁版嵁鍏ュ簱锛氬皢 CSV 鏀惧叆 `data/ATLAS_800_A3_752T_128G_DIE/hccl/v8.5/`锛堝搴?`communication_data_ref: "../../hccl/v8.5/"`锛?| HDY | 3.14 | ProfilingDataSource `_lookup_comm` 鑳藉懡涓?|
| C10-3 | 鍐掔儫楠岃瘉锛歚pytest tests/perf_database/ -k comm -v` | HDY | 3.14 | 閫氫俊鏌ヨ鍗曞厓娴嬭瘯閫氳繃 |
| H3 | HCCL Test 浜ゅ弶楠岃瘉锛氱敤 hccl_test 宸ュ叿瀵圭浉鍚?message_sizes 璺戜竴閬嶏紝涓?Python benchmark 瀵规瘮 | HDY | 3.18 | 鍋忓樊 <10%锛涢噸鐐归獙璇?1MB/256MB/512MB 寮傚父鐐?|
| C10-4锛堝彲閫夛級| tier=0锛坕nter_pod锛夋暟鎹噰闆嗭細闇€澶氳妭鐐癸紙>16 鍗★級鐜 | HDY | 瑙嗚祫婧?| 鏈夊鑺傜偣璧勬簮鏃惰ˉ閲?|

### 鏁版嵁鍏ュ簱璺緞

```
tensor_cast/performance_model/perf_database/data/
鈹斺攢鈹€ ATLAS_800_A3_752T_128G_DIE/
    鈹溾攢鈹€ vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5/
    鈹?  鈹斺攢鈹€ op_mapping.yaml  鈫?communication_data_ref: "../../hccl/v8.5/"
    鈹斺攢鈹€ hccl/
        鈹斺攢鈹€ v8.5/            鈫?鏂板缓鐩綍锛屾斁 4 涓?CSV
            鈹溾攢鈹€ hcom_allReduce_.csv
            鈹溾攢鈹€ hcom_allGather_.csv
            鈹溾攢鈹€ hcom_reduceScatter_.csv
            鈹斺攢鈹€ hcom_alltoallv_.csv
```

### 寮傚父鍊煎鐞嗙瓥鐣?

閲嶆柊閲囬泦鍚庤嫢浠嶆湁寮傚父鍊硷紙鍗曟娴嬮噺鎶栧姩锛夛紝澶勭悊浼樺厛绾э細
1. **H3 浜ゅ弶楠岃瘉**锛氱敤 hccl_test 纭鐪熷疄鍊硷紝浠?hccl_test 缁撴灉涓哄噯瑕嗙洊寮傚父琛?
2. **InterpolatingDataSource**锛氬紓甯稿€间細琚彃鍊煎钩婊戯紝瀵圭鍒扮绮惧害褰卞搷鏈夐檺
3. **tier=0 缂哄け**锛氬綋鍓?DSV3 TP=4 EP=8 鐨?all_to_all 璧?tier=0锛屾殏鏃?fallback analytic锛岀瓑澶氳妭鐐硅祫婧?

---

## 6. Phase 2锛氭暟鎹墿鍏?+ 铻嶅悎 Pass + DSV3 娣卞害鍖归厤锛?.16-3.20锛? 涓伐浣滄棩锛?

**鐩爣**锛歁icrobenchmark 鏁版嵁鎵╁厖 + KvRmsNormRopeCache Pass + DSV3 MoE/MLA 鍖归厤 + Attention 鎻掑€煎崌绾?+ DSV3 mini 楠岃瘉銆?

---

### 浠诲姟 E锛歁icrobenchmark + Attention 鍗囩骇锛圱CX锛?

**鍙傝€?*锛氳璁℃枃妗?搂6.1-搂6.4锛堟暟鎹簱鏋勫缓涓夋璧帮級銆伮?.2锛堣绠楃畻瀛?Microbenchmark锛夈€伮?.4锛團usedAttention Microbenchmark锛?

| # | 妫€鏌ョ偣 | 瀹屾垚鏃ユ湡 | 楠屾敹鏍囧噯 |
|---|-------|---------|---------|
| E1 | `generate_shape_grid.py`锛氭寜 kernel_type 鍒嗘淳鐢熸垚閫昏緫锛圙EMM: 妯″瀷 N/K + M 缃戞牸; Attention: 妯″瀷 heads + batch脳seq 缃戞牸; Elementwise: 妯″瀷 hidden + num_tokens 缃戞牸锛? powers-of-2 琛ュ厖 | 3.16 | Qwen3 + DSV3 shape 缃戞牸瑕嗙洊瀹為檯缁村害 |
| E2 | `microbenchmark generation tool`锛氳 op_mapping.yaml 鐨?torch_npu_reference 鐢熸垚鑴氭湰 | 3.17 | 鐢熸垚鐨勮剼鏈娉曟纭?|
| E3 | 闆嗙兢 Microbenchmark 閲囬泦 + `database build tool` | 3.19 | 姣忎釜 kernel_type CSV 琛屾暟 > Profiling 鍘熷 |
| E4 | FusedAttention Microbenchmark锛堟瀯閫?paged KV cache 杈撳叆锛?| 3.20 | FIA CSV 瑕嗙洊澶氱 (batch_size, seq_len) 缁勫悎 |
| E5 | Attention 鎻掑€?sqrt 鍙樻崲锛歄(n^2) 绠楀瓙鎻掑€煎墠鍋?sqrt 绾挎€у寲 | 3.20 | 涓嶅悓 seq_len 涓?FIA 鎻掑€艰宸?<20% |

---

### 浠诲姟 F锛欿vRmsNormRopeCache Pass锛圶JT锛?

**鐩爣**锛氬疄鐜?KvRmsNormRopeCache 铻嶅悎 pass锛屼娇 TC dispatch trace 涓?DSV3 Profiling 涓殑 `KvRmsNormRopeCache` kernel 瀵归綈銆?

**鑳屾櫙**锛欴SV3 Decode 涓?`KvRmsNormRopeCache` 鍗?0.8%锛?501 娆¤皟鐢級銆俆C 褰撳墠灏嗗叾鍒嗚В涓?`rms_norm` + `apply_rope` + `reshape_and_cache` 涓変釜鐙珛 op銆侼PU 鏈夊搴旂殑铻嶅悎 kernel `npu_kv_rmsnorm_rope_cache`锛坥p-plugin 宸叉湁鏉＄洰锛夈€?

**鍙傝€冨疄鐜?*锛?
- 妯″紡鍙傝€冿細`compilation/patterns/rms_norm.py`锛?44 琛岋紝RmsNorm 绫?pattern锛? `patterns/rotary_embedding.py`锛?1 琛岋級
- 鍥炬搷浣滃弬鑰冿細`compilation/freezing_passes/grouped_matmul_swiglu_pass.py`锛?04 琛岋級
- 鑷畾涔?op 娉ㄥ唽锛歚ops/mla.py`锛堝凡鏈?mlapo op锛屾柊澧?kv_rms_norm_rope_cache锛?

**淇敼鑼冨洿**锛氭柊澧?`compilation/patterns/kv_rms_norm_rope_cache.py`锛屼慨鏀?`compilation/patterns/__init__.py`锛屾柊澧?op 鍒?`ops/mla.py`

| # | 妫€鏌ョ偣 | 瀹屾垚鏃ユ湡 | 楠屾敹鏍囧噯 |
|---|-------|---------|---------|
| F1 | KvRmsNormRopeCache pattern + custom op + 娉ㄥ唽 | 3.17 | 鍗曞厓娴嬭瘯 + dispatch trace 鍑虹幇 `tensor_cast.kv_rms_norm_rope_cache` |

**宸ヤ綔閲忎及绠?*锛殈160 琛屼唬鐮侊紝2-3 澶┿€?

**MoeGatingTopK**锛歈1 涓嶅仛 pass锛岀敤 op_mapping composite 鎴?analytic fallback 鍏滃簳銆俀2 琛?pass锛堥浼?250 琛岋紝3-4 澶╋級銆?

---

### 浠诲姟 G锛歁oE/MLA 鍖归厤锛圸H锛?

**鍙傝€?*锛氳璁℃枃妗?搂4.2锛坈omposite 鏌ヨ + MLA 鍒嗚В锛?

**鍓嶇疆渚濊禆**锛歁oE/MLA spec锛圸H璧疯崏 3.12锛孒XW review锛?

| # | 妫€鏌ョ偣 | 瀹屾垚鏃ユ湡 | 楠屾敹鏍囧噯 |
|---|-------|---------|---------|
| G1 | MoE 绠楀瓙鍖归厤锛歁oeGatingTopK, MoeDistributeDispatch/CombineV2 | 3.18 | 鍗曞厓娴嬭瘯 |
| G2 | MLA 鍒嗚В鏌ヨ瀹屽杽锛氬尯鍒?Prefill/Decode 瀛愬唴鏍?shape锛堣璁℃枃妗?搂4.2 MLA 鍒嗚В琛級 | 3.20 | 鍗曞厓娴嬭瘯 |

---

### 浠诲姟 H锛欴SV3 娣卞害鍒嗘瀽 + 楠岃瘉宸ュ叿锛圸ZY + HDY + TCX锛?

| # | 璐熻矗浜?| 妫€鏌ョ偣 | 瀹屾垚鏃ユ湡 | 楠屾敹鏍囧噯 |
|---|-------|-------|---------|---------|
| H1 | HDY | DSV3 Decode Profiling 閫愬眰鑰楁椂鍒嗘瀽 | 3.16 | 鍒嗘瀽鎶ュ憡 |
| H2 | ZZY | TC vs Profiling 绠楀瓙瀵归綈琛紙Qwen3 + DSV3锛?| 3.18 | 瀵归綈琛ㄦ牸锛堝尮閰?涓嶅尮閰?鍘熷洜锛?|
| H3 | HDY | HCCL Test 浜ゅ弶楠岃瘉 | 3.18 | Python benchmark 涓?hccl_test 鍋忓樊 <10%锛堣 搂C10 鍚庣画璁″垝锛?|
| H4 | ZZY | 鏈鐩栫畻瀛愬垎鏋?+ 鑰楁椂褰卞搷璇勪及 | 3.20 | 缂哄彛娓呭崟 + 浼樺厛绾ф帓搴?|
| H5 | TCX | `database validation tool`锛氶€愮畻瀛?+ 绔埌绔簿搴︽姤鍛婅緭鍑?| 3.20 | 绮惧害鎶ュ憡鍙緭鍑?|

### Phase 2 妫€鏌ョ偣锛?.19 Review + 3.20 鏀跺熬锛?

| 浜や粯鐗?| 楠屾敹鏍囧噯 | 璐熻矗浜?|
|-------|---------|--------|
| 鎵╁厖 CSV 鏁版嵁搴?| shape 瑕嗙洊 > Profiling 鍘熷 | TCX |
| FIA Microbenchmark + sqrt 鎻掑€?| 澶氱 batch/seq + 璇樊 <20% | TCX |
| database validation tool | 绮惧害鎶ュ憡鍙緭鍑?| TCX |
| KvRmsNormRopeCache Pass | 鍗曞厓娴嬭瘯閫氳繃 | XJT |
| DSV3 MoE/MLA 鍖归厤 | 鍗曞厓娴嬭瘯閫氳繃 | ZH |
| DSV3 瀵归綈鍒嗘瀽 | 瀵归綈琛ㄦ牸 + 缂哄彛娓呭崟 | ZZY + HDY |

**DSV3 Mini 楠岃瘉**锛?.19锛夛細鍚?Phase 1 鏍煎紡锛岃鐩?DSV3 Decode 鍦烘櫙銆?

---

## 7. Phase 3锛氱鍒扮绮惧害楠岃瘉锛?.19-3.23锛? 涓伐浣滄棩锛?

**鐩爣**锛氱鍒扮绮惧害 <15%锛屼氦浠樼簿搴︽姤鍛娿€?

> **璇存槑**锛歅hase 3 涓?Phase 2 灏鹃儴鏈?1 澶╅噸鍙狅紙3.19-3.20锛夛紝ZH鍜孹JT鍙湪 3.19 Phase 2 Review 鍚庣洿鎺ュ惎鍔ㄧ鍒扮楠岃瘉銆?

| # | 妫€鏌ョ偣 | 瀹屾垚鏃ユ湡 | 璐熻矗浜?| 楠屾敹鏍囧噯 |
|---|-------|---------|--------|---------|
| J1 | Qwen3-32B 绔埌绔獙璇侊紙Prefill + Decode锛?| 3.20 | ZH + XJT | 璇樊 <15%, 瑕嗙洊 >90% |
| J2 | DSV3 绔埌绔獙璇侊紙Decode, MoE + MLA锛?| 3.20 | HDY + XJT | 璇樊 <15%, 瑕嗙洊 >90% |
| J3 | 绮惧害闂瀹氫綅 + 淇 | 3.23 | ZZY鍒嗘瀽 + ZH/TCX淇 | 琛ユ暟鎹?淇槧灏?璋冩彃鍊?|
| J4 | 绮惧害鎬绘姤鍛?| 3.23 | 鍏ㄥ憳 | 浜や粯 |

---

## 8. 椋庨櫓涓庣紦瑙?

| # | 椋庨櫓 | 褰卞搷 | 姒傜巼 | 缂撹В鎺柦 |
|---|------|------|------|---------|
| R1 | 闆嗙兢璧勬簮涓嶈冻 | E3/C10 寤惰繜 | 涓?| 3.10 鍓嶉绾︼紱Phase 1 鐢ㄧ幇鏈?Profiling 鏁版嵁 |
| R2 | KvRmsNormRopeCache Pass 姣旈鏈熷鏉?| F1 寤舵湡 | 浣?| op_mapping composite 鍏滃簳锛涙湁 RmsNorm+RoPE 鐜版垚 pattern 鍙傝€?|
| R3 | DSV3 MoE/MLA 鏄犲皠澶嶆潅 | G1/G2 寤舵湡 | 涓?| 绌垮埡宸查獙璇侀€氱敤閫昏緫锛汬DY鍏ㄨ亴 DSV3 鍒嗘瀽闄嶄綆涓嶇‘瀹氭€?|
| R4 | Attention 鍖归厤绮惧害涓嶈冻 | 绔埌绔宸秴鏍?| 涓?| FIA.csv 宸叉湁 67 琛岋紱E4 琛?Microbenchmark锛汦5 sqrt 鎻掑€?|
| R5 | 绔埌绔簿搴?<15% 闅捐揪鍒?| Phase 3 璋冧紭鏈熶笉瓒?| 楂?| **鏍稿績缂撹В**锛歅hase 1/2 鍚勫仛 mini 楠岃瘉鎻愬墠鏆撮湶闂 |
| R6 | op_mapping 閿欒鑷寸郴缁熸€?MISS | 鍖归厤鐜囦笅闄?| 涓?| 鍙屼汉浜ゅ弶 review锛沝iscover_operators 妫€娴嬭鐩栫巼 |
| R7 | 閫氫俊鍗犳瘮楂樹絾绮惧害涓嶈冻锛圦wen3 89.8%锛?| Qwen3 璇樊瓒呮爣 | 涓?| Phase 1 mini 楠岃瘉纭 CommAnalytic 绮惧害 |
| R8 | ZH 50% 瀵艰嚧 Phase 2 DataSource 杩涘害涓嶈冻 | G1/G2 寤舵湡 | 涓?| TCX鎵挎媴 attention+鎻掑€煎噺杞籞H璐熸媴锛汳oE/MLA spec 鎻愬墠鍑嗗 |
| R9 | XJT琚叾浠栭」鐩嫋浣?| A2 寤舵湡褰卞搷鍏ㄩ槦 | 涓?| A2 鏄叏闃熻В閿佺偣锛?.9纭杩涘睍锛涘繀瑕佹椂 SE 鍏滃簳 |

---

## 闄勫綍 A锛氬垎鏀瓥鐣?

```
develop (绋冲畾涓荤嚎)
  |
  +-- feat/perf-database (浠?db-spike 鍒涘缓)
        |
        +-- XJT: feat/perf-db-compiler
        +-- ZH:   feat/perf-db-datasource
        +-- TCX: feat/perf-db-toolchain
        +-- ZZY: feat/perf-db-op-mapping
        +-- HDY: feat/perf-db-op-mapping-dsv3
```

**鎿嶄綔姝ラ**锛?
- Step 1锛?.6锛孒XW锛夛細鍒涘缓 feat/perf-database锛岀‘璁ゆ祴璇曢€氳繃
- Step 2锛?.6-3.9锛屽悇璐熻矗浜猴級锛氭媺涓汉鍒嗘敮
- Step 3锛?.23锛孒XW锛夛細feat/perf-database PR 鍥?develop

---

## 闄勫綍 B锛氬弬鑰冪储寮?

姣忎釜浠诲姟娑夊強鐨勮璁℃枃妗?鏁欑▼绔犺妭閫熸煡锛?

| 浠诲姟 | 璁捐鏂囨。绔犺妭 | 鍏朵粬鍙傝€?|
|------|------------|---------|
| A1-A2 CLI | 搂5.1-搂5.3 | 鈥?|
| A3 铻嶅悎 Pass | 搂9.1 | `compilation/freezing_passes/patterns/matmul_allreduce.py` |
| B1 閫氫俊鏌ヨ | 搂4.2, 搂4.4, 搂4.7 | `examples/comm_config_example.yaml` |
| B2 Composite | 搂4.2 (composite + MLA 鍒嗚В) | `performance_model/__init__.py` (shape 鎺ㄥ) |
| C1-C8 op_mapping | 搂4.5, 搂7.1-搂7.2 | `tutorial/OP_PLUGIN_MAPPING_TUTORIAL.md`, `examples/op_mapping_example.yaml` |
| C9-C10 HCCL | 搂6.3 | HCCL Test 鏂囨。 |
| D1 瑙ｆ瀽 | 搂6.5 | 鈥?|
| D2 Attention | 搂4.8 | 绌垮埡鎶ュ憡 搂4.1 |
| D3 鎻掑€?| 搂4.4 | AI Configurator `perf_database.py` 鎻掑€兼柟娉?|
| D4 鍙戠幇 | 搂6.6 | 鈥?|
| E1-E4 Microbench | 搂6.1-搂6.4 | 鈥?|
| F1 KvRmsNormRopeCache | 搂9.1 | `compilation/patterns/rms_norm.py`, `patterns/rotary_embedding.py` |
| G1-G2 MoE/MLA | 搂4.2 | 鈥?|

---

## 闄勫綍 C锛氳繘灞曠鐞嗙粏鍒?

### 椋炰功鏃ユ姤

姣忎汉姣忓ぉ 18:00 鍓嶆洿鏂帮紝妯℃澘锛?

```
銆愭棩鎶ャ€戝鍚?鏃ユ湡

瀹屾垚锛?
- [浠诲姟 ID] 鍏蜂綋瀹屾垚鍐呭

杩涜涓細
- [浠诲姟 ID] 杩涘睍鎻忚堪

闃诲锛?
- 鏃?/ 鎻忚堪闃诲鍘熷洜鍜岄渶瑕佽皝甯姪

椋庨櫓淇″彿锛?
- 鏃?/ 鎻忚堪鍙戠幇鐨勬綔鍦ㄩ棶棰?

鏄庢棩璁″垝锛?
- [浠诲姟 ID] 璁″垝鍋氫粈涔?
```

**瑙勫垯**锛?
- "闃诲"= 鎴戞棤娉曠户缁帹杩涳紝闇€瑕佸閮ㄥ府鍔?
- "椋庨櫓淇″彿"= 鎴戣兘缁х画浣嗗彂鐜颁簡娼滃湪闂
- 杩炵画 2 澶╁悓涓€浠诲姟鏃犺繘灞曚笖鏃犻樆濉烇紝SE 涓诲姩璇㈤棶

### DIMA 鐪嬫澘

鎸?Phase 鍒?Swimlane锛屾瘡涓鏌ョ偣涓€寮犲崱鐗囥€傚繀濉瓧娈碉細Owner銆丏ue Date銆丼tatus锛圱o Do / In Progress / Review / Done / Blocked锛夈€?

### 绔欎細瑙勫垯

- 涓ユ牸 15 鍒嗛挓锛屾瘡浜?2 鍒嗛挓
- **鍙洖绛斾袱涓棶棰?*锛?) 鏈夐樆濉為渶瑕佸府鍔╁悧锛?) 鍙戠幇椋庨櫓淇″彿浜嗗悧锛?
- 鎵€鏈変汉閮藉洖绛?鏃?鈫?3 鍒嗛挓鏁ｄ細

### Review 鑺傜偣

| 鏃堕棿 | 褰㈠紡 | 鍐呭 |
|------|------|------|
| **3.13** | Review 浼?1h | Phase 1 mini 绔埌绔粨鏋?+ Phase 2 浼樺厛绾ц皟鏁?|
| **3.19** | Review 浼?1h | DSV3 mini 楠岃瘉 + Phase 3 go/no-go |
| **3.23** | Review 浼?1h | 绮惧害鎶ュ憡 Review |

