# V25 Gate E Report (`V25_GATE_E_DEBUG40`)

**Overall:** `PASS`  
**Execution:** `exec_d774fd58d6dc`  
**Authority:** `debug_only`

## Gates
- Data: PASS — {'train_rgb': 40, 'train_thermal': 40, 'val_rgb': 15, 'val_thermal': 15, 'pairing_errors': 0, 'missing_images': 0, 'illegal_bbox': 0, 'unknown_category': 0, 'duplicate_sample_ids': 0, 'pair_audit_ok': True}
- Model: PASS — queries=300 size=[640, 640]
- Artifacts: PASS
- Metrics: PASS — mAP50_95=0.4010 mAP50=0.5037 preds@0.1=944.0 changed=True
- Resources: peak_gpu_mb=832.90771484375 duration_s=336.50687408447266 exec_dir_mb=1510.56

## Next
Freeze Gate E and proceed to Gate F real RGB-T dataset registration (read-only + one-batch only).
