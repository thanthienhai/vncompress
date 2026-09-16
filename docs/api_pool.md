# Gọi API song song nhiều key

**Module:** `vncompress/api_pool.py` · **Smoke test:** `scripts/probe_api_pool.py` · **Cấu hình:** `.env`

---

## 1. Vấn đề

Một key trên endpoint này có ~50 request/phút. Một lần chạy stage 3 đầy đủ là hàng chục nghìn call — một key biến việc 2 tiếng thành việc 2 ngày.

Bốn key không chỉ nhân 4 thông lượng. Chúng **tách rời lỗi**: một key bắt đầu trả 429 sẽ tự nghỉ, ba key còn lại vẫn chạy, thay vì cả run đứng lại.

## 2. Cái gì tính theo từng key

| | Phạm vi | Vì sao |
|---|---|---|
| Rate limit | **theo key** | provider đếm theo key. Một limiter dùng chung sẽ hoặc bóp 4 key xuống hạn mức của 1 key, hoặc tệ hơn, cho 4 key cùng dồn vào quota của 1 key |
| Concurrency | **theo key** | 10 request đồng thời mỗi key. Đây là cái chặn burst trở thành 429 |
| Cool-down | **theo key** | một key 429 không phải là cả run hỏng. Nghỉ riêng key đó chính là lý do giữ 4 key |
| Retry | **theo ITEM** | việc mà key đang nghỉ không làm xong sẽ quay lại hàng đợi và key khoẻ nhặt lên → cool-down tốn thời gian, không bao giờ tốn độ phủ |

`4 key × 10 = 40 worker`, trần ~200 req/phút.

## 3. Chính sách lỗi

| Mã | Xử lý | Vì sao |
|---|---|---|
| **429** | nghỉ key đó **cố định 60s** (+jitter ≤25%), item quay lại hàng đợi | 429 chỉ nói một điều: key này vượt hạn mức phút. Hạn mức nạp lại theo đồng hồ cố định, nên chờ đúng một cửa sổ là đủ. Backoff nhân đôi ở đây là backoff chống lại một cái đồng hồ đã cho sẵn đáp án — sau 4 lần 429 nó sẽ treo một key khoẻ mạnh suốt 8 phút để chờ một cửa sổ 60 giây |
| **401 / 403** | **vô hiệu hoá key vĩnh viễn, ngay lập tức** | key bị thu hồi trả 401 tức thì trên mọi call. Coi nó là retryable thì nó rút cạn hàng đợi ở tốc độ tối đa và làm hỏng mọi item — cách nhanh nhất để mất cả run |
| **mất kết nối** (`RemoteProtocolError`, `ConnectError`, `ReadError`, `PoolTimeout`) | nghỉ key **3s**, item quay lại hàng đợi | Đây là 499 mà không kịp được cấp số. Gateway ngắt trước khi có response, nên `openai` ném `APIConnectionError` **không mang status** — không có HTTP response nào để đặt số vào. Phân loại theo "có số nguyên hay không" thay vì theo bản chất lỗi chính là thứ đã vứt 131/1284 call (~25%) trong một batch, mỗi call bỏ sau đúng 1 lần thử. Lặp lại luôn an toàn: *"without sending a response"* nghĩa là request chưa từng được trả lời, và một call nén không có side effect nào để nhân đôi |
| 400, 404, 422, 5xx | bỏ item, không retry | lỗi ở request hoặc ở server, lặp lại chỉ lặp lại lỗi |

**Mọi lỗi đều ghi vào `provenance/api_errors.jsonl`**, kể cả 429 (có nhãn riêng). Nhờ đó sau khi chạy xong có thể phân biệt "run chậm" với "run hỏng" từ file, chứ không phải từ những gì đã trôi qua màn hình.

Mọi cool-down đều có jitter ≤25%: 4 key cùng 429 trong một giây không được phép cùng quay lại trong một giây rồi 429 tiếp.

Nếu **mọi** key đều chết, `run_pool` ném `PoolExhausted` chứ không trả về kết quả rỗng — một run mất hết key mà báo thành công trông y hệt một run hoàn thành với rất ít dữ liệu.

## 4. Cấu hình

`.env` (đã gitignore; `.env.example` là template được commit):

```bash
VNCOMPRESS_TEACHER_BASE_URL=https://token-api.fpt.ai/v1
VNCOMPRESS_TEACHER_MODEL=GLM-5.2
VNCOMPRESS_TEACHER_API_KEY_1=...
VNCOMPRESS_TEACHER_API_KEY_2=...
VNCOMPRESS_TEACHER_API_KEY_3=...
VNCOMPRESS_TEACHER_API_KEY_4=...

VNCOMPRESS_JUDGE_MODEL=Llama-3.3-70B-Instruct   # khác họ với teacher
VNCOMPRESS_RPM=50
VNCOMPRESS_CCU_PER_KEY=10
```

Key **phải khác nhau**. Cùng một key dán vào hai slot là **một** hạn mức mang hai tên — pool sẽ trông như được cấp 4x rồi đâm thẳng vào giới hạn. Trùng lặp bị loại tự động.

## 5. Smoke test trước mỗi lần chạy dài

```bash
python scripts/probe_api_pool.py --requests 120
```

Tốn 120 call, trả lời 4 câu mà nếu không hỏi bây giờ thì 3 tiếng nữa mới biết:

1. Mọi key trong `.env` có thật sự chạy không? Key bị thu hồi không phân biệt được với key tốt cho đến khi dùng.
2. Việc có trải đều 4 key không, hay đã âm thầm dồn về một key?
3. Rate limit phía client có giữ không? Có 429 ở đây nghĩa là hạn mức thật thấp hơn `.env` khai.
4. Lỗi trả về hình dạng thế nào? `classify()` phải đọc đúng status, nếu không cả chính sách retry lệch hết.

Kết quả đo được (2026-09-15, 120 request): 120/120 thành công, chia **30/30/30/30** tuyệt đối, 0 lỗi, median 1.04s.

> Với **20** request trên 40 worker, chỉ key1 và key2 được dùng — 20 thread đầu đã vét sạch hàng đợi trước khi thread của key3/key4 kịp lấy. Đó không phải lỗi; probe tự in cảnh báo và gợi ý chạy lại với số request ≥ 3× số worker.

## 5b. Mất kết nối: cái đã vứt 25% số call

**Triệu chứng.** Một batch 1.284 item mất 131 call (~25%), mỗi call bỏ sau đúng 1 lần thử. Log ghi 131 dòng giống hệt nhau:

```
APIConnectionError: Connection error.
```

Bốn chữ đó là **toàn bộ** những gì `openai` nói, và nó giống nhau dù socket bị từ chối, gateway ngắt giữa chừng, DNS hỏng hay TLS hỏng. Sửa đầu tiên không phải sửa lỗi mà là **sửa log**: ghi cả cause chain.

```
APIConnectionError: Connection error.
  <- RemoteProtocolError: Server disconnected without sending a response.
```

**Đây là 499 không kịp được cấp số.** Gateway ngắt trước khi có response, nên không có HTTP response nào để đặt status vào. Phân loại theo "có số nguyên hay không" thay vì theo bản chất sự kiện chính là thứ đã vứt 131 mẫu đã trả tiền. Lặp lại luôn an toàn: *"without sending a response"* nghĩa là request chưa từng được trả lời, và một call nén không có side effect nào để nhân đôi.

**Ba giả thuyết, đo từng cái:**

| Giả thuyết | Phép đo | Kết quả |
|---|---|---|
| Quá nhiều connection so với `httpx.Limits` | nâng `max_connections` theo số worker | vẫn lỗi |
| Concurrency quá cao | sweep 5/10/15/20 per key, payload thật, 32k max_tokens | **0 lỗi ở mọi mức** — 80 concurrent chạy sạch |
| Socket cũ trong pool bị gateway thu hồi | tắt hẳn tái dùng (`max_keepalive_connections=0`) | **vẫn 48% lỗi** — giả thuyết sai |

Điểm mấu chốt: sweep chạy **burst ngắn** thì sạch, còn run **kéo dài** thì hỏng. Khác biệt không nằm ở số connection mà ở thời gian.

**Sai lầm phải tránh khi sửa.** Bản fix đầu cho mỗi lần mất kết nối "cool key 3s" — tái dùng nhánh 499. Nhưng cool-down **park cả 20 worker của key đó**. Với ~40 lỗi/phút chia 4 key, mỗi key nằm không ~35s trong mỗi 60s: throughput rơi từ 64 xuống 10 item/phút. Đúng nguyên tắc đã viết ở §3 — 429 là lỗi của key nên phạt key, socket chết thì **không phải lỗi của key** — nên giờ item quay lại hàng đợi mà key không bị đụng tới (`DISCONNECT_COOLDOWN = 0.0`).

## 5c. Hai bug làm mất dữ liệu âm thầm

**Ctrl-C bị nuốt.** `run_pool` bắt `KeyboardInterrupt` rồi return bình thường, nên vòng lặp batch không biết gì: nó ghi batch dở dang vào manifest là `done` rồi chạy tiếp batch sau. Mọi lần resume về sau **bỏ qua** phần chưa chạy. Đã xảy ra thật: batch 1 ghi `done` với 392/500 row. Giờ interrupt được ném lại, batch ghi `partial`, và shard vẫn được lưu (những row đó đã trả tiền rồi).

**Worker chết vì BaseException làm treo run.** Item của nó biến mất, `done` không bao giờ bằng `total`, các thread còn lại quay vòng trên hàng đợi rỗng. Nhìn từ ngoài giống hệt API chậm. Giờ item được trả lại hàng đợi trước khi exception lan ra.

## 6. Bẫy đã gặp với GLM-5.2 (reasoning model)

Đây là thứ suýt làm hỏng cả run, và không lộ ra ở `--dry-run`.

**`max_tokens` bao gồm cả token suy luận, và suy luận chạy trước.** Đo thực tế:

| Cấu hình | `finish_reason` | tokens | nội dung trả về |
|---|---|---|---|
| `max_tokens=16`, thinking on | `length` | 42 | **rỗng** |
| `max_tokens=512`, thinking on | `stop` | 499 | `"Hà Nội."` |
| `max_tokens=64`, **thinking off** | `stop` | **24** | `"Hà Nội"` |

Ba hệ quả đã xử lý:

1. **Tắt thinking mặc định** (`extra_body={'thinking': {'type': 'disabled'}}`). Nén ở đây là viết lại trích xuất theo ngân sách cứng — không có gì để chain-of-thought suy luận, nên nó chỉ tốn hệ số ~20x token. `--thinking` để bật lại.
2. **`max_tokens = 32768`, phẳng.** Bản trước tính theo `target_tokens`; đo trên 9 call pilot thật thì **4/9 bị `finish_reason='length'`**. Một bản nén bị cắt giữa chừng thì **ngắn**, nên nó đọc như trượt ngân sách, bị re-prompt 2 lần, rồi mẫu bị loại — cả run sẽ báo "teacher không đạt ngân sách" trong khi thực tế là nó chưa bao giờ được viết hết câu. `max_tokens` là **trần, không phải chỗ đặt trước** — tính tiền theo token thực sinh ra, nên để cao không tốn gì khi không dùng tới, còn cắt cụt thì mất cả mẫu.
3. **Bắt `finish_reason == 'length'` thành lỗi riêng** (`TruncatedResponse`), không requeue — key khác cũng sẽ cắt y hệt. Ghi thành lý do loại riêng để báo cáo phân biệt được "teacher trượt ngân sách" với "ta chưa cho nó viết xong".

**Judge cũng vậy:** `gpt-oss-120b` là reasoning model và trả về nội dung rỗng ở `max_tokens=64`. Judge mặc định là `Llama-3.3-70B-Instruct` — khác họ với GLM (điều kiện bắt buộc) **và** không reasoning. Một judge mà câu trả lời có thể biến mất vào ngân sách suy luận là một judge chấm hàng "không trả lời được" vì lý do chẳng liên quan gì tới hàng đó.
