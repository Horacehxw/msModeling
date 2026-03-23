# zero_cost 鍒嗙被瑙勫垯

## 姒傝堪

`zero_cost: true` 琛ㄧず璇ョ畻瀛愬湪 NPU 涓婃棤 kernel 鎵ц,杩斿洖 0.0 us銆傚垎涓轰袱绫?

1. **褰㈢姸绠楀瓙**: view, permute, reshape 绛?鈥?TC 鍜?NPU 閮戒笉鎵ц
2. **TC 鍒嗚В浼奖**: TC 灏嗚瀺鍚堢畻瀛愭媶涓哄瓙姝ラ,浣?NPU 宸插皢瀛愭楠よ瀺鍚堝埌鍏朵粬 kernel

## 绫诲埆 1: 褰㈢姸绠楀瓙 (Shape-only ops)

NPU 鏃犳暟鎹Щ鍔?绾厓鏁版嵁鎿嶄綔:

```yaml
aten.view.default:           zero_cost: true
aten.permute.default:        zero_cost: true
aten.t.default:              zero_cost: true
aten.transpose.int:          zero_cost: true
aten._unsafe_view.default:   zero_cost: true
aten.unsqueeze.default:      zero_cost: true
aten.split.Tensor:           zero_cost: true
aten.split_with_sizes.default: zero_cost: true
aten.select.int:             zero_cost: true
aten.slice.Tensor:           zero_cost: true
aten.detach.default:         zero_cost: true
aten.alias.default:          zero_cost: true
aten.expand.default:         zero_cost: true
```

**鍒ゆ柇鏍囧噯**: Profiling 涓案杩滀笉鍑虹幇瀵瑰簲 kernel Type 鈫?zero_cost銆?

## 绫诲埆 2: TC 鍒嗚В浼奖 (Decomposition artifacts)

TC 灏?NPU 铻嶅悎绠楀瓙鍒嗚В涓哄涓瓙姝ラ銆傚瓙姝ラ鐨勫欢杩熷凡鍖呭惈鍦ㄨ瀺鍚堢畻瀛愪腑,閲嶅璁＄畻浼氬鑷撮珮浼般€?

### MoE 璺敱鍒嗚В 鈫?MoeGatingTopK 铻嶅悎

NPU 鐨?`MoeGatingTopK` 铻嶅悎浜?
- softmax + top-k + weight normalization + mask 鎿嶄綔

TC 灏嗗叾鍒嗚В涓?
```
moe_gating_topk(logits, bias, k)  鈫?鏈夋槧灏?鎹曡幏铻嶅悎寤惰繜
  鈫?鐒跺悗 TC 缁х画鍒嗚В璺敱閫昏緫:
aten.topk()           鈫?宸茶 MoeGatingTopK 鍖呭惈 鈫?zero_cost
aten.sum.dim_IntList() 鈫?宸茶 MoeGatingTopK 鍖呭惈 鈫?zero_cost
aten.sigmoid()        鈫?shared expert gate,negligible 鈫?zero_cost
aten.where.self()     鈫?鏉′欢閫夋嫨,宸茶瀺鍚?鈫?zero_cost
aten.bitwise_not()    鈫?mask 鍙嶈浆,TC 鍐呴儴 鈫?zero_cost
```

### MoE FFN 鍒嗚В 鈫?DispatchFFNCombine 铻嶅悎

NPU 鐨?`DispatchFFNCombine` 铻嶅悎浜?
- InitRouting + Dispatch + 2脳MatMul + SwiGlu + Combine + Unpermute

TC 灏嗗叾鍒嗚В涓虹嫭绔嬬畻瀛?褰?DFC pass 涓嶇敓鏁堟椂):
```
permute_tokens     鈫?鏈夋槧灏?(MoeDistributeDispatchV2, alternate: DFC)
grouped_matmul脳N   鈫?鏈夋槧灏?(GroupedMatmul, alternate: DFC)
swiglu             鈫?鏈夋槧灏?(SwiGlu)
unpermute_tokens   鈫?鏈夋槧灏?(MoeDistributeCombineV2, alternate: DFC)
```
娉ㄦ剰: 杩欎簺瀛愮畻瀛愪繚鎸佺嫭绔嬫槧灏?涓嶆槸 zero_cost),鍥犱负:
- DFC pass 鐢熸晥鏃?瀹冧滑涓嶅嚭鐜?鈫?鏃犲奖鍝?
- DFC pass 涓嶇敓鏁堟椂(褰撳墠 TC 瀹炵幇),闇€瑕佺嫭绔嬫煡璇?

### NPU 闆舵嫹璐?concat

```yaml
aten.cat.default:          zero_cost: true
tensor_cast.cat.default:   zero_cost: true
```

ConcatD 鍦?CANN 8.5 profiling 涓粠鏈嚭鐜般€侼PU concat 閫氬父鏄浂鎷疯礉 view 鎿嶄綔銆?

## 鍐崇瓥娴佺▼

```
Q: 璇ョ畻瀛愮殑 kernel Type 鍦?profiling 涓嚭鐜拌繃鍚?
鈹?
鈹溾攢 浠庢湭鍑虹幇
鈹?  鈹溾攢 鏄舰鐘?view 鎿嶄綔? 鈫?zero_cost (绫诲埆 1)
鈹?  鈹溾攢 寤惰繜宸插寘鍚湪鍙︿竴涓瀺鍚堢畻瀛愪腑? 鈫?zero_cost (绫诲埆 2)
鈹?  鈹斺攢 鏄嫭绔嬭绠椾絾缂烘暟鎹? 鈫?淇濇寔 kernel_type 鏄犲皠,绛夋暟鎹噰闆?
鈹?
鈹斺攢 鍑虹幇杩?鈫?涓嶆槸 zero_cost,闇€瑕?kernel_type 鏄犲皠
```

## 楠岃瘉鏂规硶

纭 zero_cost 鍒嗙被鐨勮瘉鎹?
1. 鍦ㄦ墍鏈?profiling kernel_details.csv 涓悳绱㈣ Type 鈫?纭鏈嚭鐜?
2. 杩借釜 vllm-ascend 婧愮爜,纭璇ユ搷浣滆鍝釜铻嶅悎 kernel 鍚告敹
3. 鍦?notes 瀛楁璁板綍: 琚摢涓瀺鍚堢畻瀛愬寘鍚?profiling 鏁版嵁鏉ユ簮

