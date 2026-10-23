# การแก้ `config.yaml` ตาม brief และผลกระทบ

วันที่ 2026-10-23 · branch `hands-on/task01` · อ้างอิง `DEV_AGENT_LOOP.md` หัวข้อ *Environment notes*

## สรุป

เดิมทุก profile ใช้ `deepseek-r1:1.5b` ตัวเดียว ตอนนี้แก้ให้ตรงกับ brief แล้ว
`reasoning` และ `fallback` เปลี่ยนไปใช้ `qwen3:4b` ส่วน `fast` คงเดิม
นอกจากนี้ได้แก้การให้คะแนนใน `scripts/agent_eval.py` เพื่อกันคะแนนสูงเกินจริงที่เกิดจาก `qwen3:4b`

## สิ่งที่แก้

### 1. `config.yaml`

| Profile | ค่า | ก่อน | หลัง |
|---|---|---|---|
| `fast` | ทั้งหมด | `deepseek-r1:1.5b`, `num_predict` 512, `think: false` | ไม่เปลี่ยน |
| `reasoning` | `model` | `deepseek-r1:1.5b` | `qwen3:4b` |
| | `num_predict` | 1024 | 2048 |
| | `think` | false | false |
| `fallback` | `model` | `deepseek-r1:1.5b` | `qwen3:4b` |
| | `num_predict` | 512 | 4096 |
| | `think` | false | **true** |

ค่า `temperature`, `num_ctx`, `keep_alive` และ `runtime` ไม่เปลี่ยน
คอมเมนต์บรรทัดบนสุดของไฟล์ก็แก้ให้ตรงกับ model ที่ใช้จริงแล้ว

### 2. `scripts/agent_eval.py`: ตัดส่วนที่ model คิดออกก่อนตรวจคำตอบ

เพิ่มฟังก์ชัน `final_text()` ให้ `grade()` ตรวจเฉพาะข้อความหลัง `</think>` ตัวสุดท้าย
ถ้าไม่มี `</think>` จะใช้ข้อความทั้งหมดเหมือนเดิม
(หลัง Fix 1 ใน `DEV_FIXES_01.md` ฟังก์ชันนี้กลายเป็นด่านที่สอง เพราะ M1 ตัดออกให้ตั้งแต่แรกแล้ว)

**เหตุผล:** ตอนทดสอบ `qwen3:4b` ที่ตั้ง `think: false` ข้อความที่ model คิดก่อน `</think>` ยังติดมาใน `content`
ถ้าตัวเลขที่คาดไว้ไปโผล่ในส่วนที่คิด แต่คำตอบสุดท้ายผิด ระบบเดิมก็ยังนับว่าถูก
การแก้นี้อยู่ใน eval script เท่านั้น ไม่ได้แตะ loop (`harness/agent.py`)
ตามที่ brief กำหนดว่าการให้คะแนนต้องแยกจาก loop

## การตรวจสอบ

| ตรวจอะไร | ผล |
|---|---|
| `python -m unittest discover -s tests` | 27/27 ผ่าน (ชื่อ model ใน test อ่านจาก config เอง) |
| `python scripts/check_setup.py` | ทั้ง 3 profile พบ model ใน `ollama list` |
| `agent_eval.py --profiles reasoning --only m01` | จบเป็น `done` ใน 2 step, ใช้ 96 วินาที, ไม่มี step ไหนโดนตัด, คำตอบบอกยอดเกิน 60,000 ถูกต้อง แต่ไม่ได้เรียก `calculator` จึงได้ **FAIL** ตามเกณฑ์ `expected_tools` |
| `fallback` ถาม "2+2" หนึ่งครั้ง | ตอบ `4` โดยแยกส่วนที่คิดไปไว้ใน field `thinking` ถูกต้อง แต่ใช้ 420 token และ **49 วินาที** |

## ผลกระทบ

### กระทบงานเดิมใน Session 2

- `hands_on2_eval.py` และ `tool_call_eval.py` ใช้ `fast,reasoning` เป็นค่าเริ่มต้น
  ถ้ารันตอนนี้ ฝั่ง `reasoning` จะเป็น `qwen3:4b` แทน `deepseek-r1:1.5b`
- **log เก่าเทียบกับ log ใหม่ไม่ได้ตรงๆ** จึงแยกชื่อไฟล์ตาม session ไว้ให้แล้ว:
  - ผลของ Session 2 (ทำตอนที่ทุก profile เป็น 1.5B) เปลี่ยนชื่อเป็น `logs/session2_calls.jsonl`
    (เฉพาะแถววันที่ 2 และ 10 ต.ค.), `logs/session2_hands_on2_results.csv` และ `logs/session2_tool_call_results.csv`
  - งานใหม่ใช้ชื่อ `session3_*`: `config.yaml` → `logging.file: logs/session3_calls.jsonl`
    (แถวทดสอบ agent loop วันที่ 23 ต.ค. ย้ายมาอยู่ไฟล์นี้), `session3_agent_results.csv`,
    `session3_agent_steps.csv` และ `session3_prefix_check.csv`
  - ถ้ารัน script ของ Session 2 ซ้ำ ผลจะออกเป็น `hands_on2_results.csv` / `tool_call_results.csv` (ไม่มี prefix)
    จึงไม่ทับไฟล์ `session2_*`
  - ไฟล์ log ทั้งหมดไม่อยู่ใน git (`.gitignore`)
- ใน `calls.jsonl` ให้แยกผลด้วยคอลัมน์ `model` ไม่ใช่ `profile` เพราะชื่อ profile เดิมตอนนี้ชี้ไปที่ model อื่นแล้ว

### Profile `fallback` (`think: true`, `num_predict: 4096`)

- **ช้ามาก:** คำถามง่ายๆ ยังใช้เกือบ 1 นาที
- token ที่ใช้คิดนับรวมใน `num_predict` ถ้าคิดยาวเกินไปจะโดนตัด (`done_reason = length`)
- `ask_structured()` ใน Hands-on 2 จะตกมาใช้ profile นี้หลัง `fast` และ `reasoning` ล้มเหลวหมด
  การเปิดโหมดคิดพร้อมกับ structured output (`format=schema`) อาจได้ JSON ที่ไม่เสถียร
  ถ้าเห็นอาการนี้ ให้ดูว่าเกิดจาก fallback หรือไม่ก่อนจะแก้โค้ด

### เวลาและหน่วยความจำ

- `agent_eval.py` แบบเต็ม (10 งาน × 2 profile) คาดว่าใช้ 30–60 นาที
  (`qwen3:4b` ใช้ประมาณ 1.5–4 นาทีต่องาน)
- ทั้งสอง model ตั้ง `keep_alive: 10m` จึงอาจโหลดค้างพร้อมกัน 2 ตัว
  ถ้า RAM หรือ VRAM ไม่พอ Ollama จะสลับ model เข้าออก ทำให้ `load_s` สูงขึ้นและ prefix cache หาย

### Prefix check

- `prefix_check.py` ใช้ `reasoning` (`qwen3:4b`) เป็นค่าเริ่มต้นแล้ว ซึ่งเรียก tool ได้จริง จึงน่าจะเห็นความต่างของ cache ชัดกว่าตอนใช้ 1.5B
- ระหว่างรัน ห้ามเรียก model อื่น (รวมถึง `agent_eval.py`) เพราะจะล้าง cache

### ปัญหาที่ยังเหลือ

- ~~**`qwen3:4b` ยังคิดออกมาใน `content` ทั้งที่ตั้ง `think: false`**~~ **แก้แล้วใน M1** (`DEV_FIXES_01.md` Fix 1):
  `split_thinking()` ใน `harness/llm_client.py` ย้ายส่วนที่คิดไปไว้ใน `ChatResult.thinking`
  ส่วนนี้จึงไม่ถูกส่งกลับเข้า history ไม่ทำให้ completion check ปัดตก และไม่ไปอยู่ใน answer ที่บันทึก
  แต่ละบรรทัด `llm_call` ใน log มี `thinking_chars` ไว้ดูว่ายังมีการคิดหลุดมาหรือไม่
  **ข้อที่ยังเป็นอยู่:** `think: false` ไม่ได้หยุด `qwen3:4b` จากการคิด model ยังใช้เวลาและ token ขาออกไปกับส่วนนี้
  จึงยังเสี่ยงโดนตัด (`done_reason = length`) ส่วนการเปลี่ยนไปใช้ tag ของ qwen3 ที่ไม่คิดเป็นเรื่องที่ต้องตัดสินใจใน `config.yaml`
- **`check_setup.py`:** test call ของ `fast` ได้คำตอบว่าง (`''`) เพราะ `num_predict=16` น้อยเกินสำหรับ `deepseek-r1:1.5b`
  ปัญหานี้มีอยู่ก่อนแล้วและ profile `fast` ไม่ได้ถูกแก้

README แก้ให้ตรงกับ config แล้ว (model ที่ต้อง pull, ตาราง profile, 1.5B กับ 4B, ชื่อไฟล์ log)

## วิธีย้อนกลับ

```bash
git checkout 4c4a408 -- Harness-01/config.yaml          # กลับไปใช้ deepseek-r1:1.5b ทุก profile
git checkout 4c4a408 -- Harness-01/scripts/agent_eval.py  # เอาการตัด </think> ออก
```
