---
name: project-status
description: Generate perf-database project progress dashboard. Default=concise console view; ALL=full PDF report; TCX/ZH/HDY/etc=person-focused view
---

# 绠楀瓙鎬ц兘鏁版嵁搴?鈥?椤圭洰杩涘睍鐪嬫澘

涓?SE 鐢熸垚鍙鍖栬繘搴︽姤鍛婏紝鏁村悎浠ｇ爜銆佹枃妗ｃ€佹棩鎶ャ€佺珯浼氱邯瑕佺瓑澶氱淮淇℃伅婧愩€?

## 鍙傛暟瑙ｆ瀽

| 璋冪敤鏂瑰紡 | 琛屼负 |
|---------|------|
| `/project-status` (鏃犲弬鏁? | **绠€鍖栫増**: 鐩存帴鍦?console 杈撳嚭銆傝仛鐒︽暣浣撹繘搴︺€乀OP 椋庨櫓銆侀渶瑕?SE 鍏虫敞鐨勫喅绛栫偣銆佷笅涓€姝ュ缓璁€倊200 琛屼互鍐?|
| `/project-status ALL` | **瀹屾暣鐗?*: 鐢熸垚瀹屾暣椤圭洰鐪嬫澘锛屼繚瀛樺埌 `docs/perf_database/daily_project_status/杩涘睍鐪嬫澘_{YYYYMMDD}.md` (鍚屾椂澶囦唤鍒?`/Users/horacehxw/Documents/hxw-鍗庝负/铓傝殎浠跨湡鍣ㄩ」鐩?AI杩涘睍鎬荤粨\`)銆傛牸寮忕編瑙傦紝闈㈠悜鍏ㄤ綋椤圭洰鎴愬憳 |
| `/project-status <浜哄憳浠ｅ彿>` | **涓汉鐗?*: 鍦?console 杈撳嚭銆傚寘鍚叏灞€鎽樿 + 璇ユ垚鍛樼殑涓撻」鍏虫敞浜嬮」銆侀樆濉炪€佷笅涓€姝ュ缓璁€備唬鍙? TCX/ZH/ZZY/HDY/LJW/DSH/QCX/HXW |

## 鏁版嵁閲囬泦姝ラ

**鎵€鏈夋ā寮忓叡鐢?*銆備娇鐢?Agent 宸ュ叿骞惰閲囬泦浠ヤ笅 5 绫讳俊鎭簮銆?

### Step 1: 骞惰閲囬泦 (鍚姩 3-4 涓?Agent)

**Agent 1: 浠ｇ爜浠撳簱鍒嗘瀽**
```
1. git fetch --all
2. git log --oneline -20 feat/perf-database (涓荤嚎杩涘睍)
3. git branch -r --list 'gitcode/*' | 閫愬垎鏀粺璁￠鍏?钀藉悗 feat/perf-database 鐨?commit 鏁?
4. git log --oneline gitcode-ascend/develop --not feat/perf-database (涓讳粨鏂板)
5. pytest tests/perf_database/ --ignore=tests/perf_database/test_reference_data_e2e.py -q (娴嬭瘯鐘舵€?
6. git diff --stat remotes/gitcode/develop..feat/perf-database | tail -3 (鍙樻洿瑙勬ā)
```

**Agent 2: 鏂囨。鍒嗘瀽 (Design Doc 涓哄悎瑙勫熀鍑?**
```
1. 璇诲彇 docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v1.4.md
   - 搂4 鏍稿績妯″潡: 閫愭ā鍧楁鏌ュ疄鐜?vs spec 鍋忓樊
   - 搂7.5 璇勪及鎸囨爣: M1-M6 瀹炵幇鐘舵€?
   - 搂8 寮€鍙戣鍒? Phase 浜や粯鐗?vs 瀹為檯瀹屾垚
   - 搂9.1 铻嶅悎 Gap: 鍚勯」鐘舵€?
   - 搂1.4 楠屾敹鏍囧噯: E2E <15%, 瑕嗙洊 >90%, 宸ュ叿閾?9 涓?
2. 璇诲彇 docs/perf_database/WORK_PLAN_Q1.md (v3.2)
   - 鎵€鏈夋鏌ョ偣鐨勫畬鎴愮姸鎬?(鉁?馃攧/鈴?
   - Phase 鏃堕棿绾?vs 褰撳墠鏃ユ湡
   - 椋庨櫓琛?R1-R14
3. 璇诲彇鏈€鏂扮殑 E2E 楠岃瘉鎶ュ憡 (reports/phase1-e2e-*/phase1_e2e_*_verification_report_zh.md)
   - M1-M6 鎸囨爣鏁版嵁
   - MISS 鏍瑰洜鍒嗙被
   - Phase 2/3 TODO 浼樺厛绾?
4. 璇诲彇 CHANGELOG (docs/perf_database/CHANGELOG_*.md)
5. 璇诲彇 docs/perf_database/METRICS_GUIDE.md (M1-M6 鎸囨爣瀹氫箟鍜岀敤娉?
```

**Agent 4: M1-M6 鎸囨爣璁＄畻** (濡傛灉 profiling 鏁版嵁鍙敤)
```
杩愯 4 涓満鏅殑 TC profiling 鍛戒护锛岄噰闆?M1-M6 鍏ㄩ噺鎸囨爣:

DATA_DIR="$(pwd)/tensor_cast/performance_model/perf_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.15.0_torch2.9.0_cann8.5"

# M1-M5 (鍦ㄧ嚎): 4 涓満鏅悇璺戜竴娆?TC profiling + --export-metrics

# Qwen3 Prefill: --enable-flashcomm-v1 瀵规爣 vLLM ENABLE_FLASHCOMM1=1
# FlashComm 灏?all_reduce鈫抮ms_norm 鏇挎崲涓?reduce_scatter鈫抮ms_norm鈫抋ll_gather
# 鏁堟灉: M3 33%鈫?0%, M5 62%鈫?8%, 浣?reduce_scatter CSV 鏈夎啫鑳€椋庨櫓 (35ms/call vs 鐪熷疄 ~2ms)
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 10 --query-length 4104 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile --perf-database "$DATA_DIR" \
  --enable-flashcomm-v1 \
  --export-metrics results/qwen3_prefill_metrics.json --log-level info

# Qwen3 Decode: 鏆備笉鍔?--enable-flashcomm-v1
# 鍘熷洜: vLLM cudagraph FULL_DECODE_ONLY 妯″紡涓?decode 鍙兘涓嶈蛋 flashcomm 璺緞
# decode profiling 涓槸鍚︽湁 hcom_reduceScatter_ 寰呯‘璁? 濡傛湁鍒欏姞姝?flag
python3.10 -m tensor_cast.scripts.text_generate Qwen/Qwen3-32B \
  --num-queries 16 --query-length 1 --context-length 4096 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 16 \
  --quantize-linear-action DISABLED \
  --performance-model profiling --compile --perf-database "$DATA_DIR" \
  --export-metrics results/qwen3_decode_metrics.json --log-level info

# DSv3 Prefill: 瀵规爣 profiler-dsv3-input2048-output1 (QBM batch=2048)
# 鍙傛暟鎺ㄥ: profiling kernel_details 涓?QuantBatchMatmulV3 batch dim=2048
# vLLM max-num-batched-tokens=2048, 鍗曡姹?ISL=2048 鈫?nq=1 ql=2048
python3.10 -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --num-queries 1 --query-length 2048 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 8 --dp-size 2 --ep-size 16 \
  --quantize-linear-action W8A8_STATIC \
  --performance-model profiling --compile --perf-database "$DATA_DIR" \
  --export-metrics results/dsv3_prefill_metrics.json --log-level info

# DSv3 Decode: 瀵规爣 profiler-dsv3-input4096-output1536-concurrency8 (QBM batch=5)
# 鍙傛暟鎺ㄥ: profiling kernel_details 涓?QuantBatchMatmulV3 batch dim=5
# vLLM max-num-seqs=8, DP=2 鈫?per-rank ~5 queries 鈫?nq=10 (dp_size=2, 10/2=5)
python3.10 -m tensor_cast.scripts.text_generate deepseek-ai/DeepSeek-V3 \
  --num-queries 10 --query-length 1 --context-length 4096 --word-embedding-tp row \
  --device ATLAS_800_A3_752T_128G_DIE --world-size 16 --tp-size 8 --dp-size 2 --ep-size 16 \
  --quantize-linear-action W8A8_STATIC \
  --performance-model profiling --compile --perf-database "$DATA_DIR" \
  --export-metrics results/dsv3_decode_metrics.json --log-level info

# M6 (鍗婄绾?: TC Prediction Ratio = tc_full_prediction / real_per_fwd
# M6=1.0 瀹岀編, >1 楂樹及, <1 浣庝及. Phase 3 鐩爣: 0.85-1.15
#
# M6 璁＄畻鏂规硶璁?(2026-03-20 淇):
#   鍒嗗瓙: TC 鍏ㄩ噺棰勬祴 (empirical + analytic fallback), 浠?--export-metrics JSON 鐨?
#         m6_input.tc_predicted_total_s 鑾峰彇
#   鍒嗘瘝: 鐪熷疄鍗曚釜 forward pass 鑰楁椂, 閫氳繃浠ヤ笅姝ラ鑾峰彇:
#     1. 浠?kernel_details.csv 涓瘑鍒?forward pass 杈圭晫 (瀵绘壘姣?forward pass 鍑虹幇鎭板ソ涓€娆＄殑
#        anchor kernel, 濡?ArgMaxV2/ApplyTopKTopPCustom 绛?sampling kernel, 鎴?
#        DispatchFFNCombine/FusedInferAttentionScore 绛夋ā鍨嬬壒鏈?kernel)
#     2. AI 鍒嗘瀽姣忎釜 forward pass 鐨勭粨鏋?(batch dim, kernel 缁勬垚)
#     3. 纭鎵€鏈?forward pass 缁撴瀯涓€鑷村悗, 鐢?Stage / n_forward_passes 浣滀负鍒嗘瘝
#     4. 濡備笉涓€鑷? 闇€鎵惧埌涓?TC 浠跨湡鍙傛暟鍖归厤鐨勯偅涓€涓?forward pass
#
# TC 鍙傛暟鎺ㄥ鏂规硶璁?
#   1. 浠?profiling kernel_details.csv 鐨?QuantBatchMatmulV3 (涓昏绠?kernel) 璇?batch dim
#   2. 鏍规嵁 batch dim 鍜?DP/TP 閰嶇疆鍙嶆帹 TC 鐨?nq/ql 鍙傛暟
#   3. 鐢?anchor kernel 鍒囧垎 forward passes, 楠岃瘉姣忎釜 pass 缁撴瀯鏄惁涓€鑷?
#
# Qwen3 profiling 鏁版嵁 (0314, 鏉冨▉鏉ユ簮):
PROF_QWEN3="/Users/horacehxw/Data/Profiling/Profiling-0317-full/profiler-qwen3-0314"
# DSv3 profiling 鏁版嵁 (0319, 鏉冨▉鏉ユ簮):
PROF_DSV3="/Users/horacehxw/Data/Profiling/Profiling-0320-DSv3/profiler-dsv3-0319"
# 娉ㄦ剰: 涓嶅啀浣跨敤 Profiling-0313-phase1-e2e-test (鏃у熀绾?, Qwen3/DSv3 缁熶竴鐢ㄤ笂闈㈢殑璺緞

# Qwen3 M6: 浣跨敤 0314 profiling 鏁版嵁
python3.10 tools/perf_data_collection/compute_m6.py \
  --tc-report results/qwen3_prefill_metrics.json \
  --profiler-output "$PROF_QWEN3/profiler-qwen3-input4096-output1"

python3.10 tools/perf_data_collection/compute_m6.py \
  --tc-report results/qwen3_decode_metrics.json \
  --profiler-output "$PROF_QWEN3/profiler-qwen3-input4096-output1536-concurrency4-rrate2"

# DSv3 M6: 浣跨敤 0319 profiling 鏁版嵁
python3.10 tools/perf_data_collection/compute_m6.py \
  --tc-report results/dsv3_prefill_metrics.json \
  --profiler-output "$PROF_DSV3/profiler-dsv3-input2048-output1"

python3.10 tools/perf_data_collection/compute_m6.py \
  --tc-report results/dsv3_decode_metrics.json \
  --profiler-output "$PROF_DSV3/profiler-dsv3-input4096-output1536-concurrency8-rrate4"

# 姹囨€?M1-M5
python3.10 -c "
import json
print('鍦烘櫙                  M1     M2     M3     M4     M5     TC(ms)')
print('-' * 70)
for name, path in [
    ('Qwen3 PF (+FC)', 'results/qwen3_prefill_metrics.json'),
    ('Qwen3 DC',       'results/qwen3_decode_metrics.json'),
    ('DSv3 PF',        'results/dsv3_prefill_metrics.json'),
    ('DSv3 DC',        'results/dsv3_decode_metrics.json'),
]:
    r = json.load(open(path))
    m1=r['m1']['m1_raw_op_count_hr']
    m2=r['m2']['m2_fused_op_hr']
    m3=r['m3']['m3_fused_op_hr_no_zc']
    m4=r['m4']['m4_per_shape_hr']
    m5=r['m5']['m5_simulated_latency_coverage']
    tc=r['m6_input']['tc_predicted_total_s']*1000
    print(f'{name:<22} {m1:>5.1%} {m2:>5.1%} {m3:>5.1%} {m4:>5.1%} {m5:>5.1%} {tc:>8.1f}')
"

# ==========================================
# M6 璁＄畻: AI 鍒嗘瀽 profiling forward pass
# ==========================================
# compute_m6.py 浼氳嚜鍔ㄥ鎵?anchor kernel (榛樿 ArgMaxV2) 鍋?forward pass 鍒囧垎骞跺彇骞冲潎銆?
# 浣?**蹇呴』** 鐢?AI (鍗充綘) 鍦ㄤ娇鐢ㄥ墠鍏堥獙璇?
#
# 姝ラ 1: 鍒嗘瀽 profiling 涓瘡涓?forward pass 鐨勭粨鏋?
#   瀵规瘡涓?profiling 鍦烘櫙, 鐢?kernel_details.csv 鍋氫互涓嬪垎鏋?
#   - 璇嗗埆 anchor kernel 鍋?forward pass 杈圭晫鍒囧垎 (姣?fwd 鎭板ソ鍑虹幇涓€娆＄殑 kernel type)
#   - 妫€鏌ユ瘡涓?forward pass 鐨?kernel 缁勬垚 (DFC/QBM/FIA/RING_MLA 鏁伴噺)
#   - 鎻愬彇姣忎釜 forward pass 鐨?QuantBatchMatmulV3 batch dim (= 瀹為檯 batch size)
#   - 纭: 鎵€鏈?forward pass 缁撴瀯鏄惁涓€鑷?(batch dim, kernel 鏁伴噺, 绠楀瓙绫诲瀷)
#
# 姝ラ 2: 鍖归厤 TC 浠跨湡鐨?forward pass
#   - 濡傛灉鎵€鏈?forward pass 缁撴瀯涓€鑷?鈫?鍙互鐢?Stage/N 浣滀负 M6 鍒嗘瘝
#   - 濡傛灉涓嶄竴鑷?(娣峰悎 prefill+decode, 涓嶅悓 batch size):
#     鈫?鎵惧埌 batch dim 鍖归厤 TC 鍙傛暟鐨勯偅涓€涓?forward pass
#     鈫?鐢ㄨ forward pass 鐨?kernel duration sum 浣滀负 M6 鍒嗘瘝
#     鈫?娉ㄦ剰: kernel sum 鍚?compute-comm overlap, 搴斾紭鍏堢敤 step_trace Stage/N
#
# 姝ラ 3: 璁＄畻 M6
#   M6 = TC_prediction / real_per_fwd
#   - 鍒嗗瓙: m6_input.tc_predicted_total_s (娣峰悎棰勬祴: empirical for HIT + analytic for MISS)
#   - 鍒嗘瘝: AI 纭鍚庣殑鍗曚釜 forward pass 鐪熷疄鑰楁椂
#
# 绀轰緥 (宸查獙璇佺殑 forward pass 缁撴瀯):
#   DSv3 PF (input2048): 12 涓竴鑷寸殑绾?prefill pass, QBM batch=2048, Stage/12=295ms
#   DSv3 DC (c8): 62 涓竴鑷寸殑绾?decode pass, QBM batch=5, Stage/62=51ms
#   Qwen3 PF (input4096): 5 涓函 prefill pass, Stage/5=1147ms

# 鑷姩璁＄畻 (浠呭湪 AI 纭 forward pass 缁撴瀯鍚庝娇鐢?:
python3.10 -c "
import json
print()
print('鍦烘櫙                  TC(ms)   Real(ms)   M6      鍒ゆ柇')
print('-' * 65)
for name, path in [
    ('Qwen3 PF (+FC)', 'results/qwen3_prefill_m6.json'),
    ('Qwen3 DC',       'results/qwen3_decode_m6.json'),
    ('DSv3 PF',        'results/dsv3_prefill_m6.json'),
    ('DSv3 DC',        'results/dsv3_decode_m6.json'),
]:
    r = json.load(open(path))
    tc = r['tc_predicted_us']/1e3
    real = r['real_per_fwd_us']/1e3
    m6 = tc / real if real > 0 else 0
    flag = 'OK' if 0.85 <= m6 <= 1.15 else ('HIGH' if m6 > 1.15 else 'LOW')
    n_fwd = r.get('n_forward_passes', '?')
    print(f'{name:<22} {tc:>8.1f} {real:>8.1f} {m6:>6.3f}   {flag}  (N={n_fwd})')
print()
print('娉ㄦ剰: M6 鍥?microbench CSV 鑶ㄨ儉 (R10) 鏆備笉鍙俊銆侾hase 3 楠屾敹浠?M3+M5 涓轰富鎸囨爣銆?)
print('      濡傛灉 forward pass 缁撴瀯鏈粡 AI 楠岃瘉, M6 鍒嗘瘝鍙兘涓嶅噯纭€?)
"
```

娉ㄦ剰: 濡傛灉 profiling 鏁版嵁璺緞涓嶅彲璁块棶锛堝 Profiling 鏁版嵁鐩綍涓嶅瓨鍦級锛孧6 鏃犳硶璁＄畻锛岃烦杩囧苟娉ㄦ槑銆?
濡傛灉 results/*.json 宸插瓨鍦ㄤ笖鏃ユ湡涓哄綋澶╋紝鍙洿鎺ヨ鍙栬€屼笉閲嶆柊杩愯 TC銆?

**Agent 3: 鍥㈤槦鍔ㄦ€?*
```
1. 璇诲彇鏃ユ姤: /Users/horacehxw/Documents/hxw-鍗庝负/铓傝殎浠跨湡鍣ㄩ」鐩?鏃ユ姤\鏃ユ姤姹囨€?txt
2. 璇诲彇鎵€鏈夌珯浼氱邯瑕?PDF: /Users/horacehxw/Documents/hxw-鍗庝负/铓傝殎浠跨湡鍣ㄩ」鐩?鏃ユ姤/鏅鸿兘绾*.pdf
   鍏堝垪鍑烘墍鏈夌邯瑕佹枃浠讹紝鐒跺悗閫愪釜鎻愬彇:
   ls "/Users/horacehxw/Documents/hxw-鍗庝负/铓傝殎浠跨湡鍣ㄩ」鐩?鏃ユ姤/"鏅鸿兘绾*.pdf
   瀵规瘡涓?PDF 浣跨敤 text 妯″紡鎻愬彇 (浣?context 寮€閿€):
   python3.10 ~/.claude/scripts/read_pdf.py "<pdf_path>" --mode text
   濡傞渶鏌ョ湅鍥捐〃/娴佺▼鍥? 鏀圭敤 image 妯″紡:
   python3.10 ~/.claude/scripts/read_pdf.py "<pdf_path>" --mode image --pages 1
   鐒跺悗鐢?Read 宸ュ叿璇诲彇杈撳嚭鐨?PNG 鏂囦欢
   - 鎻愬彇姣忎汉杩涘睍銆侀樆濉炪€侀闄╀俊鍙?
   - 鎻愬彇绔欎細鍐崇瓥鍜?action items
   - 鎻愬彇寰呭姙娓呭崟
3. 浜ゅ弶楠岃瘉: 鏃ユ姤璇寸殑 vs 绔欎細绾 vs 浠ｇ爜瀹為檯鍙樻洿
```

### Step 2: 浜ゅ弶鍒嗘瀽

瀵归噰闆嗗埌鐨勬暟鎹繘琛屼氦鍙夐獙璇?
1. **瀹炵幇 vs Design Doc**: 姣忎釜宸插畬鎴愪换鍔℃槸鍚︾鍚?spec锛熷亸宸嵆涓洪闄?
2. **璁″垝 vs 瀹為檯**: Work Plan 鐨勬鏌ョ偣鏃ユ湡 vs 瀹為檯瀹屾垚鏃ユ湡锛岃瘑鍒欢鏈熻秼鍔?
3. **鏃ユ姤 vs 浠ｇ爜**: 鏃ユ姤璇?瀹屾垚"浣嗕唬鐮佹湭鍚堝叆锛熸棩鎶ユ湭鎻愬埌浣嗕唬鐮佹湁鍙樻洿锛?
4. **鎸囨爣瓒嬪娍**: M1-M6 褰撳墠鍊?vs Phase 鐩爣锛屽樊璺濆垎鏋愩€傞噸鐐瑰叧娉?
   - M3 vs >50% 鐩爣 (Phase 2)
   - M5 vs >80% 鐩爣 (Phase 3)
   - M6 vs 0.85-1.15 鐩爣 (Phase 3): M6<1 璇存槑瑕嗙洊涓嶈冻锛堥渶鏇村 microbench 鏁版嵁锛夛紝M6>1 璇存槑 microbench 鍋忛珮
   - M4 MISS shape list 鈫?鎸囧 microbench 鏁版嵁閲囬泦浼樺厛绾?

### Step 3: 鎸夋ā寮忕敓鎴愯緭鍑?

---

## 杈撳嚭妯℃澘: 绠€鍖栫増 (榛樿, console)

鐩存帴鍦ㄥ綋鍓?console 鍥炲锛屼笉鐢熸垚鏂囦欢銆傛牸寮忓涓?

```markdown
# 椤圭洰鐘舵€侀€熻 | {YYYY-MM-DD}

## 杩涘害: Phase {N} {鐘舵€亇 | 璺濅氦浠?{X} 澶?
{涓€鍙ヨ瘽褰撳墠鐘舵€亇

## 鏃堕棿绾?
{ASCII 鏃堕棿绾垮浘锛屾爣娉?Phase 璧锋銆佸綋鍓嶄綅缃€侀噷绋嬬}
绀轰緥:
Phase 1 [3.6鈹佲攣鈹佲攣鈹佲攣鈹?.13] 鉁?GO
Phase 2 [3.16鈹佲柖鈹佲攣鈹佲攣3.20]   鈫?浠婂ぉ鍦ㄨ繖閲?
Phase 3      [3.19鈹佲攣鈹佲攣3.23]
浜や粯                    3.23 馃幆

## 鍏抽敭鎸囨爣 (M1-M6)
{瀹屾暣 6 鎸囨爣琛?+ 杩涘害鏉
绀轰緥:
| 鍦烘櫙 | M1 | M2 | M3 | M4 | M5 | M6 (ratio) |
|------|:--:|:--:|:--:|:--:|:--:|:----------:|
| Qwen3 PF | 77.8% | 47.4% | 23.1% | 47.4% | 52.4% | 0.531 |
| Qwen3 DC | 84.1% | 63.2% | 46.2% | 63.2% | 59.5% | 1.607 |
| DSv3 PF  | 70.6% | 50.0% | 15.4% | 41.2% | 71.9% | 0.531 |
| DSv3 DC  | 71.8% | 52.3% | 19.2% | 43.1% | 56.3% | 0.303 |

Phase 鐩爣杩涘害 (M2-M6 鍏ㄥ垪):
| 鎸囨爣 | 鐩爣 | Qwen3 PF | Qwen3 DC | DSv3 PF | DSv3 DC | 杩涘害 |
|------|:---:|:--------:|:--------:|:-------:|:-------:|------|
| M2   | (GO/NO-GO) | 47.4% | 63.2% | 50.0% | 52.3% | 鈻撯枔鈻撯枔鈻撯枒鈻戔枒鈻戔枒 |
| M3   | >50% | 23.1% | 46.2% | 15.4% | 19.2% | 鈻撯枔鈻戔枒鈻戔枒鈻戔枒鈻戔枒 |
| M4   | (璇婃柇) | 47.4% | 63.2% | 41.2% | 43.1% | 鈻撯枔鈻撯枔鈻戔枒鈻戔枒鈻戔枒 |
| M5   | >80% | 52.4% | 59.5% | 71.9% | 56.3% | 鈻撯枔鈻撯枔鈻撯枒鈻戔枒鈻戔枒 |
| M6   | 0.85-1.15 | 0.531 | 1.607 | 0.531 | 0.303 | 鈻撯枔鈻撯枒鈻戔枒鈻戔枒鈻戔枒 |

濡傛湁鍘嗗彶鏁版嵁锛岄檮瓒嬪娍:
M3 瓒嬪娍 (Qwen3 PF):  Phase1 鈫?Phase2 鈫?褰撳墠
                       31.2% 鈫??      鈫??

鎸囨爣璇存槑 (绠€):
- M1: 鍘熷 HIT 鐜?(debug 鐢? 琚?zero_cost 鑶ㄨ儉)
- M2: 铻嶅悎绠楀瓙 HIT 鐜?(GO/NO-GO 闂ㄦ, 鍚?zero_cost)
- M3: 璁＄畻绠楀瓙 HIT 鐜?(鏍稿績杩涘害, 鎺掗櫎 zero_cost) 鈥?Phase 2 楠屾敹
- M4: per-shape HIT 鐜?(缂哄彛璇婃柇, miss list 鎸囧 microbench 閲囬泦)
- M5: 浠跨湡寤惰繜瑕嗙洊 (寤惰繜鍔犳潈, 澶х畻瀛愪紭鍏? 鈥?Phase 3 楠屾敹
- M6: empirical E2E ratio (vs 鐪熷疄 per-fwd, 鐩爣 0.85鈥?.15) 鈥?Phase 3 楠屾敹

**鎽樿瑙勫垯**: 鎽樿/閫熻涓繀椤诲睍绀?**M2-M6** (5 涓寚鏍?, M1 鍥?zero_cost 鑶ㄨ儉鍙渷鐣ャ€?
瀹屾暣鐪嬫澘鐨勬寚鏍囪〃蹇呴』灞曠ず **M1-M6** 鍏ㄩ儴 6 涓寚鏍囥€?

## 椋庨櫓鐭╅樀
{鐢?ASCII 鐭╅樀鍙鍖?TOP 椋庨櫓鐨?褰卞搷脳姒傜巼 鍒嗗竷}
绀轰緥:
褰卞搷 鈫?
 楂? 鈹?R5鈼?       R11鈼?
 涓? 鈹?   R10鈼? R3  R7
 浣? 鈹?             R13
     鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈫?姒傜巼
       浣?   涓?   楂?

## TOP 椋庨櫓璇︽儏
1. 馃敶 {椋庨櫓鎻忚堪} 鈥?{褰卞搷} 鈥?{缂撹В}
2. 馃煛 ...
3. 馃煛 ...

## 闇€瑕?SE 鍐崇瓥/鍏虫敞
- {鍏蜂綋鍐崇瓥鐐癸紝濡? "閫氫俊寤烘ā鏂规 P/D 鍒嗗紑 vs 鍥哄畾寮€閿€锛孒DY 寰呭畾璁?}
- {闇€瑕佹帹鍔ㄧ殑澶栭儴渚濊禆}

## 鍥㈤槦璐熻浇涓€瑙?
{绱у噾琛ㄦ牸: 姣忎汉褰撳墠涓昏浠诲姟 + 闃诲鐘舵€?+ 渚濊禆鍏崇郴}
绀轰緥:
| 鎴愬憳 | 涓昏浠诲姟 | 鐘舵€?| 闃诲 |
|------|---------|:----:|------|
| TCX | Microbench | 馃煝 | 绛?QCX NPU 鎺ュ彛 |
| ZH  | 鏌ヨ鎺ュ彛  | 馃煝 | 鈥?|
| HDY | Profiling 閲囬泦 | 馃煛 | comm gap 1.66-20.6x |

## 鏈懆杩涘睍浜偣
- {3-5 鏉℃牳蹇冭繘灞晑

## Git 鍒嗘敮鐘舵€?
{涓庝笂娓哥殑 ahead/behind 鍙鍖杴
绀轰緥:
feat/perf-database vs gitcode-ascend/develop:
  ours 鈻堚枅鈻堚枅鈻堚枅鈻堚枅鈻堚枅鈻堚枅鈻堚枅鈻堚枅鈻堚枅鈻堚枅 +120 commits ahead
  upstream 鈻堚枅 +9 commits (闇€ rebase)

## 涓嬩竴姝ュ缓璁?(鎸変紭鍏堢骇)
1. {鏈€閲嶈鐨勮鍔▆
2. ...
3. ...
```

**鍘熷垯**: 300 琛屼互鍐呫€備繚鐣欓渶瑕?SE 鐭ラ亾鍜屽喅绛栫殑淇℃伅锛岀敤 ASCII 鍙鍖栧寮虹洿瑙傛€с€備笉灞曞紑浠诲姟鍒楄〃缁嗚妭銆?

**鍙鍖栬绱犳竻鍗?* (灏介噺鍖呭惈):
- 鏃堕棿绾垮浘 (Phase 杩涘害 + 褰撳墠浣嶇疆)
- 鎸囨爣杩涘害鏉?(鈻撯枒 鎴?鈻?椋庢牸)
- 鎸囨爣瓒嬪娍 (濡傛湁鍘嗗彶鏁版嵁)
- 椋庨櫓鐭╅樀 (褰卞搷脳姒傜巼 scatter)
- 鍥㈤槦璐熻浇琛?(鍚樆濉炵姸鎬?
- Git ahead/behind 鏉″舰鍥?

---

## 杈撳嚭妯℃澘: 瀹屾暣鐗?(ALL, 淇濆瓨鏂囦欢)

鐢熸垚瀹屾暣鐪嬫澘 Markdown锛屼繚瀛樺埌:
- **涓昏矾寰?* (git 浠撳簱鍐?: `docs/perf_database/daily_project_status/杩涘睍鐪嬫澘_{YYYYMMDD}.md`
- **澶囦唤璺緞**: `/Users/horacehxw/Documents/hxw-鍗庝负/铓傝殎浠跨湡鍣ㄩ」鐩?AI杩涘睍鎬荤粨\杩涘睍鐪嬫澘_{YYYYMMDD}.md`

**缁撴瀯** (鎸夐噸瑕佹€ф帓搴忥紝鍐崇瓥灞備俊鎭湪鍓嶏紝鎵ц缁嗚妭鍦ㄥ悗):

```markdown
# 绠楀瓙鎬ц兘鏁版嵁搴撻」鐩湅鏉?
**鏃ユ湡** | **鍒嗘敮** | **浜や粯鏃?* | **Design Doc 鐗堟湰** | **鏁版嵁鐗堟湰**

## 涓€銆佹墽琛屾憳瑕?
{3-5 鍙ヨ瘽鎬荤粨褰撳墠鐘舵€侊紝鏍稿績鐭涚浘锛屽叧閿垽鏂瓆

## 浜屻€佸叧閿寚鏍?(M1-M6)
{M1-M5: 鐧惧垎姣旀寚鏍囪〃 (4 鍦烘櫙) + M6: ratio 鍒?(1.0=瀹岀編, 鐩爣 0.85-1.15)}
{M6 闇€瑕?ASCEND_PROFILER_OUTPUT 鏁版嵁 + --export-metrics JSON锛屽涓嶅彲鐢ㄥ垯娉ㄦ槑}
{Phase 鐩爣杩涘害 + 鏀剁泭璺緞浼扮畻}

## 涓夈€乀OP 椋庨櫓 (鎸夊奖鍝嶆帓搴?
{椋庨櫓琛? #/椋庨櫓/褰卞搷/鐘舵€?缂撹В}

## 鍥涖€佹椂闂寸嚎
{ASCII 鏃堕棿绾?+ 閲岀▼纰戣〃}

## 浜斻€佹湰鍛ㄧ洰鏍囦笌浠诲姟鍒嗛厤
{P0 浜嬮」琛?+ 鍚勪汉鏈懆浠诲姟琛?(鏉ヨ嚜鏃ユ姤+绔欎細)}

## 鍏€佷笅涓€姝ュ缓璁?
{鎸変紭鍏堢骇鐨?5-6 鏉″缓璁畗

## 涓冦€丟it 鍒嗘敮鍏ㄦ櫙
{gitcode-ascend/develop 涓讳粨鏂板 commit 鍒嗘瀽 + rebase 寤鸿}
{gitcode 鍔熻兘鍒嗘敮鐘舵€佽〃 (棰嗗厛/钀藉悗/鏄惁宸插悎鍏?}
{鍏抽敭鏈悎鍏ュ垎鏀彁閱拀

## 鍏€丳hase {N} 瀹屾垚璇︽儏 (鎶樺彔)
{浠诲姟瀹屾垚娓呭崟}
{MISS 鏍瑰洜鍒嗙被}
{浠ｇ爜涓庢祴璇曠幇鐘秨
{鏃ユ姤鍏抽敭浜嬩欢鏃堕棿绾縸
```

**鏍煎紡瑕佹眰**: 闈㈠悜鍏ㄤ綋椤圭洰鎴愬憳鍒嗗彂銆備娇鐢ㄤ腑鏂囥€傝〃鏍煎榻愩€侫SCII 鍥捐〃娓呮櫚銆傞噸瑕佹暟瀛楀姞绮椼€?

---

## 杈撳嚭妯℃澘: 涓汉鐗?(<浜哄憳浠ｅ彿>, console)

鐩存帴鍦?console 鍥炲銆傚寘鍚?

```markdown
# {濮撳悕} 涓撻」鐪嬫澘 | {YYYY-MM-DD}

## 鍏ㄥ眬鐘舵€?(鎵€鏈変汉闇€鐭?
{2-3 鍙ヨ瘽: 褰撳墠 Phase, 璺濅氦浠樺ぉ鏁? 鏍稿績鐭涚浘}
{M1-M6 鎸囨爣琛?(绮剧畝, 4 鍦烘櫙)}

## 浣犵殑浠诲姟鐘舵€?
| 浠诲姟 | 鎴 | 鐘舵€?| 璇存槑 |
{浠?Work Plan + 鏃ユ姤鎻愬彇璇ヤ汉鐨勪换鍔

## 浣犵殑闃诲涓庨闄?
{浠庢棩鎶?绔欎細鎻愬彇璇ヤ汉鐨勯樆濉為」鍜岄闄╀俊鍙穧

## 渚濊禆浣犵殑涓嬫父浠诲姟
{璋佸湪绛変綘鐨勪骇鍑? 鍝簺浠诲姟渚濊禆浣犲畬鎴?}

## 浣犻渶瑕佸叧娉ㄧ殑鍗忎綔鐐?
{绔欎細 action items 涓垎閰嶇粰浣犵殑寰呭姙}
{闇€瑕佷笌璋佸榻愪粈涔坿

## 寤鸿涓嬩竴姝?(鎸変紭鍏堢骇)
1. {鏈€閲嶈}
2. ...
3. ...
```

## 浜哄憳浠ｅ彿鏄犲皠

| 浠ｅ彿 | 濮撳悕 | 鑱岃矗鍩?| Work Plan 浠诲姟 |
|------|------|--------|---------------|
| **HXW** | 璐洪獊姝?| SE, spec review + 鍐崇瓥 + 杩涘睍绠＄悊 | 鍏ㄥ眬 |
| **TCX** | 鍞愭绗?| 鏁版嵁灞? 宸ュ叿閾?+ Microbench + Attention + 鎻掑€?| D1-D4, E1-E5, H5 |
| **ZH** | 绁濊豹 | 鏌ヨ寮曟搸: _lookup_compute/comm/composite | B1-B2, G1-G2 |
| **ZZY** | 寮犻渿瀹?| Qwen3 op_mapping: BF16 楠岃瘉 + Decode + 鑷姩鍖?| C1-C5, H2, H4 |
| **HDY** | 鑳″畾涓€ | DSV3 op_mapping + HCCL + Profiling 鍒嗘瀽 | C6-C10, H1, H3 |
| **LJW** | 榄忓畤鏄?| 铻嶅悎 Pass: DispatchFFNCombine | F1-F2 |
| **XJT** | 璁搁敠娑?| (宸蹭氦鎺ョ粰 LJW) CLI + Compile Pass | A1-A3 鉁?|
| **DSH**/**CY** | 涓佷笘娴?浠庝簯) | 瀹㈡埛 database 鍒嗘敮 | 鈥?|
| **QCX** | 閽辨櫒甯?| NPU 绠楀瓙涓撳鏀寔 | 鈥?|

## 鍏抽敭鏂囦欢璺緞

| 鏂囦欢 | 鐢ㄩ€?|
|------|------|
| `docs/perf_database/OPERATOR_PERF_DATABASE_DESIGN_zh_v1.4.md` | Design Doc (鍚堣鍩哄噯) |
| `docs/perf_database/WORK_PLAN_Q1.md` | Work Plan v3.2 (浠诲姟+鏃堕棿绾? |
| `docs/perf_database/CHANGELOG_*.md` | 鍙樻洿鏃ュ織 |
| `docs/perf_database/reports/phase1-e2e-*/phase1_e2e_*_verification_report_zh.md` | E2E 楠岃瘉鎶ュ憡 |
| `/Users/horacehxw/Documents/hxw-鍗庝负/铓傝殎浠跨湡鍣ㄩ」鐩?鏃ユ姤\鏃ユ姤姹囨€?txt` | 椋炰功鏃ユ姤姹囨€?|
| `/Users/horacehxw/Documents/hxw-鍗庝负/铓傝殎浠跨湡鍣ㄩ」鐩?鏃ユ姤\鏅鸿兘绾*.pdf` | 绔欎細绾 |
| `docs/perf_database/daily_project_status/` | **涓昏緭鍑虹洰褰?* (git 浠撳簱鍐咃紝鏃ユ湡鍛藉悕) |
| `/Users/horacehxw/Documents/hxw-鍗庝负/铓傝殎浠跨湡鍣ㄩ」鐩?AI杩涘睍鎬荤粨\` | 瀹屾暣鐪嬫澘澶囦唤鐩綍 |
| `tensor_cast/performance_model/perf_database/` | 鏍稿績瀹炵幇浠ｇ爜 |
| `tools/perf_data_collection/` | 鏁版嵁閲囬泦宸ュ叿閾?|

## 娉ㄦ剰浜嬮」

- **Design Doc 鏄竴鍒囩殑鏍囧噯**: 姣忎釜妯″潡鐨勫疄鐜扮姸鎬侀兘瑕佸鐓?Design Doc spec 妫€鏌ャ€傚悎瑙勫垎鏋愪綔涓哄唴閮ㄦ柟娉曡浣跨敤锛屼笉鍗曠嫭鎴愮珷杈撳嚭鍒版姤鍛娿€備粎褰撳彂鐜?spec vs 瀹炵幇鍋忓樊鏃讹紝灏嗗亸宸啓鍏ラ闄╃珷鑺?
- **鏃ユ姤鏄潪缁撴瀯鍖栫殑**: 椋炰功鑱婂ぉ璁板綍鏍煎紡锛屽浜轰氦閿欙紝闇€瑕佷粩缁嗚В鏋愪笂涓嬫枃
- **绔欎細绾鐢?AI 鐢熸垚**: 璐ㄩ噺鍙傚樊锛岄渶涓庢棩鎶ヤ氦鍙夐獙璇?
- **gitcode-ascend/develop 鍙叧娉ㄤ富鍒嗘敮**: 鍒嗘瀽瀹冩瘮鎴戜滑澶氫簡浠€涔堝姛鑳斤紝鏄惁闇€瑕?rebase
- **gitcode 鏄垜浠洟闃熺殑 fork**: 鎵€鏈?remote 鍒嗘敮閮借 fetch 鍚庡垎鏋?
- **涓嶈鑷姩 commit**: 鐢熸垚鐨勬枃浠朵笉鍏?git
- **涓枃杈撳嚭**: 闈㈠悜涓枃鍥㈤槦

