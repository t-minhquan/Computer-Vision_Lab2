# Hướng dẫn thực hiện Bài tập 2 — Đặc trưng góc và phân vùng ảnh

- **Motivation/Background**: PDF giao nhiều thuật toán, kiểm chứng và thí nghiệm cần được tổ chức thành các bước có đầu ra rõ ràng.
- **Purpose**: Hướng dẫn nhóm tự thực hiện, đối chiếu yêu cầu và chuẩn bị bài nộp đầy đủ.
- **Overview Pipeline**: Đọc toàn bộ 9 trang PDF → đối chiếu hiện trạng → chuẩn bị → tự cài đặt → kiểm chứng → thí nghiệm → báo cáo → nộp bài.
- **Detailed Plan**: §1 Phạm vi; §2 Môi trường; §3 Notebook; §4 Phần A; §5 Phần B; §6 Thí nghiệm; §7 Báo cáo; §8 Nộp bài; §9 Rà soát.
- **References**: [PDF gốc](../../pdf/Computer_Vision%2045.pdf), [README](../../README.md), [quy ước Markdown](../../agents/rules/MD_CONVENTION.md), [quy tắc báo cáo](../../agents/rules/RESULTS_REPORTING.md).
- **Created**: 2026-11-11T00:25:41+07:00
- **Last Updated**: 2026-11-11T08:12:27+07:00

---

## 1. Phạm vi và cách sử dụng

Đây là hướng dẫn chuẩn bị và tự kiểm tra, không phải lời giải cài đặt. Các mô tả thuật toán diễn giải yêu cầu sẵn có trong PDF. Nhóm phải tự viết toàn bộ cell thuộc phần **“Cài đặt — KHÔNG dùng AI”**, kể cả sửa lỗi hay viết lại (PDF trang 2).

Các trục khảo sát bên dưới là gợi ý của PDF, không phải ý tưởng của nhóm. Trước khi nhờ AI dựng khung hay vẽ biểu đồ, nhóm cần tự quyết định ảnh, giả thuyết, tham số quét, phép đo, yếu tố giữ cố định và kết luận từ kết quả thực.

### Mục lục

- [Chuẩn bị môi trường](#2-chuẩn-bị-môi-trường)
- [Tổ chức notebook](#3-tổ-chức-notebook)
- [Phần A — Đặc trưng góc](#4-phần-a--đặc-trưng-góc)
- [Phần B — Phân vùng ảnh](#5-phần-b--phân-vùng-ảnh)
- [Thiết kế thí nghiệm](#6-thiết-kế-và-thực-hiện-thí-nghiệm)
- [Viết báo cáo](#7-viết-báo-cáo-trong-notebook)
- [Kiểm tra và nộp bài](#8-kiểm-tra-cuối-và-nộp-bài)
- [Rà soát](#9-câu-hỏi-tự-rà-soát)

| Trang PDF | Nội dung cần tra cứu |
| --- | --- |
| 1–2 | Mục tiêu, AI, thư viện, hình thức nộp |
| 3–5 | Harris, Harris–Laplace, ANMS, chữ ký hàm và kiểm chứng A |
| 6–8 | Region Growing, SRG, stale/ties, chữ ký hàm và kiểm chứng B |
| 8 | Ba yêu cầu thí nghiệm cho mỗi cặp |
| 9 | Điểm số, trường hợp 0 điểm, tài liệu tham khảo |

---

## 2. Chuẩn bị môi trường

### 2.1. Hiện trạng

Đã có [PDF](../../pdf/Computer_Vision%2045.pdf), [quy tắc](../../agents/README.md) và [README](../../README.md). Đã có [template A](../../notebooks/phan_a_harris_va_hai_bien_the.ipynb), [template B](../../notebooks/phan_b_phan_vung_anh.ipynb), môi trường `.venv` và [khai báo thư viện](../../requirements.txt). Đã có 11 hàm phần A và 4 hàm phần B; requirements.txt đã khóa phiên bản thư viện trực tiếp. Chưa có ảnh hoặc kết quả thí nghiệm. Môi trường `.venv` dùng Python 3.12.6, đã cài thư viện để kiểm tra A/B. Thông tin nhóm và phân công được quản lý tại [README](../../README.md#nhom-va-phan-cong).

### 2.2. Thiết lập trên Windows PowerShell

Mở terminal tại thư mục gốc dự án, chạy lần lượt:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy matplotlib scipy opencv-python jupyterlab ipykernel
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip freeze
.\.venv\Scripts\python.exe -m jupyter lab
```

Gọi trực tiếp Python của môi trường giúp tránh phụ thuộc thao tác kích hoạt PowerShell. Cài thư viện **không có nghĩa mọi hàm đều được phép**; quyền sử dụng từng hàm theo PDF trang 2 và [bảng trong README](../../README.md#4-quy-định-quan-trọng).

Chọn kernel thuộc môi trường vừa tạo. Ghi phiên bản Python, NumPy, SciPy, Matplotlib và OpenCV trong notebook. Sau khi môi trường chạy ổn định, lưu các phiên bản đã kiểm chứng để tái lập; đề không ấn định phiên bản thư viện cụ thể.

### 2.3. Quy ước dữ liệu

Nhóm tự thống nhất và ghi rõ:

- Ảnh xám `(H, W)`, kiểu số thực, miền cường độ cụ thể. Ngưỡng phải cùng đơn vị với ảnh.
- Điểm dùng `(row, col)`; khi vẽ, trục ngang là `col`, trục dọc là `row`.
- Cách xử lý biên ảnh, đạo hàm, ngưỡng và ghép điểm.
- Biểu diễn seeds, hạt giống hợp lệ, không trùng vị trí và định danh vùng ổn định.
- Đường dẫn tương đối từ notebook. Dùng [thư mục ảnh A](../../images/phan_a/) và [ảnh B](../../images/phan_b/) đã tạo, hiện chỉ có file giữ thư mục; nhóm tự thêm ảnh nộp theo PDF.
- Nguồn ảnh và phép chỉnh sửa; lưu riêng ảnh biến đổi, giữ ảnh gốc.

**Đầu ra bước này:** notebook mở bằng đúng kernel, đọc được ảnh, có thông tin môi trường và quy ước dữ liệu.

---

## 3. Tổ chức notebook

Đã tạo hai bản làm việc: [phần A](../../notebooks/phan_a_harris_va_hai_bien_the.ipynb) và [phần B](../../notebooks/phan_b_phan_vung_anh.ipynb). Đã cài đặt 11 hàm phần A và 4 hàm phần B; bảng số chưa có kết quả. Xem [cách mở và ghép bài](../../README.md#cấu-trúc-làm-việc-ab).

PDF yêu cầu **một notebook duy nhất** tên `bt1_<Nhom>.ipynb`. Đây là tên nguyên văn, chưa phải tệp hiện có. Vì tiêu đề là Bài tập 2, nên xác nhận điểm không nhất quán này với giảng viên trước khi đổi tên.

Đề xuất thứ tự các mục:

1. Liên kết thông tin nhóm và phân công trong README; mục tiêu, môi trường, nguồn ảnh trong notebook.
2. Chính sách AI và quy ước dữ liệu.
3. **Cài đặt — KHÔNG dùng AI: Phần A** — sinh viên tự viết.
4. Kiểm chứng Phần A, đầy đủ hình và số đo.
5. **Cài đặt — KHÔNG dùng AI: Phần B** — sinh viên tự viết.
6. Kiểm chứng Phần B, đầy đủ hình và số đo.
7. Thí nghiệm A1 so Harris.
8. Thí nghiệm A2 so Harris + `top_n`.
9. Thí nghiệm SRG so Region Growing, gồm stale và ties.
10. Kết luận, hạn chế, tài liệu tham khảo.

Mỗi cell có dùng AI cần cell Markdown **ngay phía trên** chứa nguyên văn prompt thực tế. Nếu có nhiều prompt liên quan, lưu đủ; không chỉ viết “có dùng AI” hay tóm tắt prompt. Khai báo chung đầu notebook không thay thế khai báo từng cell.

**Đầu ra:** khung mục rõ ràng, nhận diện được phần tự cài đặt và phần kiểm chứng/thí nghiệm.

---

## 4. Phần A — Đặc trưng góc

Nguồn: PDF §3, trang 3–6. Làm Harris trước vì hai biến thể dùng các thành phần của Harris.

### 4.1. Harris: kiến thức cần nắm

Hai tỉ lệ có vai trò khác nhau: `sigma_d` là độ mờ trước đạo hàm, quyết định mức chi tiết; `sigma_i` là tỉ lệ tích phân, quyết định vùng lấy trung bình các tích đạo hàm.

Công thức trong PDF:

$$
M=\sigma_D^2 G(\sigma_I)*
\begin{bmatrix}L_x^2&L_xL_y\\L_xL_y&L_y^2\end{bmatrix},
\qquad L_x=\partial_x(G(\sigma_D)*I).
$$

$$
R=\det(M)-k\,\operatorname{tr}(M)^2,\qquad k\in[0.04,0.06].
$$

Điểm góc là cực đại không gian vượt ngưỡng; không phải mọi pixel có R dương đều được giữ. R âm thường ứng với cạnh. Hệ số `sigma_d**2` nằm trong M, đặc biệt cần khi so nhiều tỉ lệ.

### 4.2. Hợp đồng hàm Harris

| Chữ ký theo PDF | Đầu ra / yêu cầu |
| --- | --- |
| `gaussian_blur(img, sigma)` | Ảnh làm mờ; được phép gọi `scipy.ndimage.gaussian_filter` cho mục đích này |
| `structure_tensor(img, sigma_i, sigma_d)` | `(Mxx, Mxy, Myy)`, mỗi mảng cùng kích thước ảnh, đã nhân `sigma_d**2` |
| `harris_response(img, sigma_i, sigma_d, k=0.04)` | Mảng R kiểu float, cùng kích thước ảnh |
| `nms_2d(response, window=3)` | Mảng bool, cực đại địa phương trong cửa sổ |
| `harris_classic(img, sigma_i=2.0, sigma_d=1.4, k=0.04, thresh_rel=0.01)` | `(N, 3)`: `(row, col, response)`, giảm dần theo response; R → NMS → ngưỡng theo `max(R)` |

**Mốc hoàn thành:** nhóm tự viết đủ hàm, giải thích được tham số, qua kiểm chứng bàn cờ trước khi dùng làm nền cho biến thể.

### 4.3. Harris–Laplace

Tháp tỉ lệ theo PDF là `sigma_n = xi**n * sigma0`. Tại mỗi tầng: `sigma_i = sigma_n`, `sigma_d = 0.7 * sigma_n`; ứng viên Harris là cực đại không gian vượt ngưỡng.

Chọn tỉ lệ bằng:

$$
|LoG(x;\sigma)|=\sigma^2\,|L_{xx}(x;\sigma)+L_{yy}(x;\sigma)|.
$$

Giữ ứng viên nếu giá trị này lớn hơn **cả hai tầng kề tại cùng vị trí** và vượt ngưỡng tối thiểu. Chọn tầng có R lớn nhất không đáp ứng yêu cầu. Không có cực trị LoG theo tỉ lệ thì phải loại điểm.

| Chữ ký theo PDF | Đầu ra / yêu cầu |
| --- | --- |
| `scale_pyramid(sigma0=1.2, xi=1.2, n_levels=10)` | Mảng sigma một chiều |
| `log_normalized(img, sigma)` | LoG đã chuẩn hóa theo công thức trên |
| `harris_laplace(img, sigmas, k=0.04, thresh_rel=0.01, log_thresh=0.0)` | `(N, 4)`: `(row, col, sigma, response)` |

Lý thuyết viết `n = 0,...,N`, còn hàm dùng `n_levels`. Nhóm cần ghi rõ số tầng, tránh nhầm với chỉ số cuối. Hai đầu tháp không có đủ hai tầng kề; quy ước xử lý phải phù hợp điều kiện cực trị của đề.

**Mốc hoàn thành:** đầu ra có sigma, thể hiện bước lọc LoG thực sự loại ứng viên; báo cáo repeatability khi phóng 3× cho cả Harris và Harris–Laplace.

### 4.4. ANMS và đường cơ sở

ANMS nhận ứng viên Harris một tỉ lệ. Bán kính theo PDF:

$$
r_i=\min_{j:\ f(x_i)<c_{robust}f(x_j)}\|x_i-x_j\|.
$$

Không có điểm mạnh hơn thỏa điều kiện thì bán kính là vô cùng. Với `c_robust=0.9`, điểm triệt tiêu phải mạnh hơn khoảng 11%. ANMS chọn theo bán kính giảm dần; `top_n` chọn theo response.

| Chữ ký theo PDF | Đầu ra / yêu cầu |
| --- | --- |
| `anms(points, n_keep, c_robust=0.9)` | `(n_keep, 3)`, giảm dần theo bán kính; đầu vào `(N, 3)` từ Harris |
| `top_n(points, n_keep)` | `(n_keep, 3)`, các điểm mạnh nhất |

Giữ cùng tập ứng viên, cùng N và đủ ứng viên cho `n_keep` khi so sánh. Ghi quy ước khi bán kính bằng nhau hoặc thiếu ứng viên; PDF chưa quy định chi tiết các trường hợp này.

**Mốc hoàn thành:** có hình điểm và hai số `grid_occupancy` với cùng N. PDF nói hàm độ phủ này “cho sẵn” nhưng không có mã/chữ ký trong tài liệu; tìm bản giảng viên cung cấp hoặc hỏi rõ định nghĩa lưới.

### 4.5. Repeatability và kiểm chứng bắt buộc

`repeatability(pts_a, pts_b, transform, img_shape, eps=3.0)` trả giá trị `[0,1]`. `transform` ánh xạ `(row, col)` từ A sang B; chỉ tính điểm có vị trí chiếu nằm trong B (PDF trang 5).

Ghi rõ kích thước ảnh đích dùng trong phép đo, quy tắc ghép điểm, trường hợp không có điểm hợp lệ và cách biến đổi ảnh. PDF không mô tả hết các chi tiết này. Nhầm ánh xạ quay/resize có thể khiến phép đo phản ánh lỗi tọa độ thay vì bộ phát hiện.

| Kiểm chứng PDF | Output cần lưu |
| --- | --- |
| Bàn cờ tổng hợp | Giao điểm trong đã biết, hình điểm phát hiện và sai số; yêu cầu ≤ 1 px |
| Quay 30° | Số điểm trước/sau và repeatability, kèm ánh xạ tọa độ |
| Phóng 3× | Hai số repeatability cho Harris và Harris–Laplace |
| So `cv2.cornerHarris` | Hình đối chiếu và mức khớp trong bán kính 2–3 px; không yêu cầu trùng tuyệt đối |
| ANMS so `top_n` | Cùng ứng viên, cùng N, cùng lưới; hai giá trị độ phủ |

Đối chiếu OpenCV chỉ ở phần được phép. Ghi tham số và quy ước làm mờ/đạo hàm trước khi giải thích khác biệt. Hoàn thành kiểm chứng có output rồi mới sang thí nghiệm mở.

---

## 5. Phần B — Phân vùng ảnh

Nguồn: PDF §4, trang 6–8. Nhóm tự viết thuật toán và hàm hỗ trợ thuộc phần cài đặt.

### 5.1. Region Growing cổ điển

Đề mô tả phát triển từ seed, xét lân cận 4, nhận pixel khi độ lệch cường độ so với trung bình vùng không vượt T. Trung bình thay đổi theo pixel được nhận, vì vậy thứ tự xử lý có thể làm đổi kết quả.

| Chữ ký theo PDF | Đầu ra / yêu cầu |
| --- | --- |
| `region_growing_classic(img, seeds, T, order="stack")` | Nhãn `int32`; `-1` cho pixel chưa thuộc vùng; hỗ trợ `stack` (LIFO), `queue` (FIFO) |

Ghi cách biểu diễn seeds, thứ tự lân cận, thứ tự xử lý nhiều seeds và thang cường độ của T. Giữ cố định các lựa chọn này khi chỉ khảo sát một yếu tố.

### 5.2. SRG

Các vùng phát triển đồng thời. Độ ưu tiên của pixel biên chưa có nhãn là độ lệch so với trung bình vùng kề; nếu kề nhiều vùng, chọn vùng cho độ lệch nhỏ nhất và đánh dấu pixel biên theo mô tả PDF. Ưu tiên độ lệch nhỏ nhất toàn cục; không dùng ngưỡng T như Region Growing.

| Chữ ký theo PDF | Đầu ra / yêu cầu |
| --- | --- |
| `srg(img, seeds, connectivity=4, stale="lazy", tie="insertion")` | `(labels, boundary_mask)`; mọi nhãn ≥ 0; dùng `heapq` |
| `srg_step_frames(img, seeds, n_frames=8, **kwargs)` | Danh sách `n_frames` ảnh nhãn tại các mốc tiến trình |
| `seed_sensitivity(img, seeds, segment_fn, n_trials=20, jitter=3)` | Bản đồ tần suất đổi nhãn khi dịch seeds trong khoảng ±jitter pixel |

Pixel biên vẫn phải được gán nhãn cuối cùng. Ghi rõ ngữ nghĩa `boundary_mask` để hình trực quan khớp điều được đo.

### 5.3. Hai chi tiết được chấm riêng

**Giá trị delta cũ:**

- `lazy`: theo PDF, khi pop phải tính lại delta; nếu khác giá trị lưu thì đưa lại hàng đợi. Giá trị tham chiếu là trung bình vùng hiện tại.
- `frozen`: delta theo cường độ hạt giống, không theo trung bình vùng đang thay đổi.

Cài cả hai và so sánh. PDF gợi ý ảnh gradient mượt; nhóm tự quyết định ảnh, phép đo và kết luận. Khi giải thích cần chỉ ra giá trị tham chiếu của delta và khả năng thích nghi với chuyển màu.

**Phá hòa:** chữ ký nêu `insertion`, `raster`, `random`. Quy tắc phải tái lập; nếu dùng `random`, cố định và ghi seed ngẫu nhiên. Cần ít nhất một ảnh mà đổi quy tắc làm đổi phân vùng. Ảnh hằng số với hai seeds là kiểm chứng bắt buộc vì mọi delta bằng nhau.

### 5.4. Kiểm chứng bắt buộc

| Kiểm chứng PDF | Output cần lưu |
| --- | --- |
| Phân hoạch đầy đủ | Điều kiện `(labels >= 0).all()` |
| Bảo toàn số vùng | Số nhãn phân biệt bằng số hạt giống hợp lệ, không trùng |
| Đảo thứ tự seeds | Hai hình và số pixel khác sau khi đối chiếu cùng định danh vùng |
| Stack / queue / SRG | Hình và số pixel khác nhau cho từng cặp |
| Ảnh phẳng, hai seeds | Hình dạng vùng, quy tắc phá hòa và nhận xét |

**Căn chỉnh nhãn:** đảo danh sách seeds có thể đổi mã nhãn dù hình phân vùng giống nhau. Quy nhãn về cùng hạt giống trước khi đếm pixel khác biệt; giữ định danh nhất quán cả trong `seed_sensitivity`. Nếu đảo seeds làm thay đổi thứ tự chèn khi có ties, cần báo rõ ảnh hưởng, không tuyên bố bất biến tuyệt đối.

Với seed jitter, ghi cách xử lý ra ngoài ảnh, trùng seed hoặc rơi sang cấu trúc khác; bản đồ là số lần đổi hay tỉ lệ, so với lần chạy gốc nào. Với frames, các mốc cần là bản chụp độc lập, không cùng tham chiếu tới trạng thái cuối.

**Mốc hoàn thành:** đủ kiểm chứng, hai chế độ stale, phá hòa tái lập và hình tiến trình.

---

## 6. Thiết kế và thực hiện thí nghiệm

Nguồn: PDF §5.1, trang 8. Cần đủ **ba cặp**, không chỉ hai phần A và B.

### 6.1. Phiếu thiết kế do nhóm tự điền

| Trường | Quyết định cần ghi trước khi chạy |
| --- | --- |
| Câu hỏi | Khác biệt cụ thể nào cần kiểm tra? |
| Giả thuyết | Dự đoán hiện tượng và bước thuật toán gây ra |
| Dữ liệu | Nguồn, kích thước, lý do chọn, chỉnh sửa |
| Biến độc lập | Yếu tố thay đổi và tập giá trị quét |
| Yếu tố cố định | Tiền xử lý, seed ngẫu nhiên, tham số khác, quy tắc ghép/đếm |
| Phép đo | Định nghĩa, đơn vị, mẫu số, cách tổng hợp |
| Bằng chứng | Hình, bảng số và đường cong sẽ lưu |

Chỉ sau khi nhóm quyết định các mục này mới dùng AI trong phạm vi đề cho phép để dựng khung/biểu đồ, kèm prompt nguyên văn. Không biến dự đoán thành kết quả đo.

### 6.2. Hướng gợi ý được PDF nêu

| Cặp | Hướng quét trong đề | Đại lượng theo gợi ý đề |
| --- | --- | --- |
| A1 so Harris | `xi`, số tầng; tỉ lệ `sigma_d/sigma_i`; `log_thresh` | Repeatability; tỉ lệ ứng viên bị loại |
| A2 so Harris + `top_n` | `c_robust` từ 0.5 đến 1.0; số điểm N | Độ phủ lưới của hai cách chọn |
| SRG so Region Growing | T; số/vị trí seeds; liên thông 4/8 | Diện tích vùng lớn nhất, số vùng, seed sensitivity |

Không bắt buộc dùng tất cả gợi ý. So `lazy`/`frozen` và ảnh hưởng phá hòa vẫn phải được bố trí rõ ở phần B, kể cả khi chọn trục quét chính khác.

### 6.3. Phép quét đạt yêu cầu

1. Giữ cố định các yếu tố ngoài trục đang xét; nếu đổi nhiều yếu tố, tách nhóm so sánh và ghi rõ.
2. Lấy đủ điểm để nhìn thấy xu hướng, bão hòa hoặc thay đổi hành vi; ba ảnh với ba tham số không thay thế đường cong.
3. Lưu cấu hình và số đo từng lần, không chỉ cấu hình tốt nhất.
4. Vẽ trục có tên, đơn vị, chú giải; giữ bảng số gốc.
5. Nếu có ngẫu nhiên, ghi seed và cách lặp/tổng hợp.
6. Giải thích bằng bước tính thay đổi; báo cáo trung thực khi kết quả khác dự đoán.

### 6.4. So sánh công bằng

- A1/Harris: cùng cặp ảnh biến đổi, phép ghép, eps; báo cả số điểm hợp lệ để hiểu đúng repeatability.
- ANMS/`top_n`: cùng ứng viên, cùng N, cùng lưới độ phủ.
- SRG/Region Growing: cùng ảnh và seeds; thể hiện cả pixel chưa gán nhãn của phương pháp cổ điển.
- Đếm pixel khác nhau: căn chỉnh định danh vùng và giữ quy ước đó qua các lần chạy.
- Thời gian: ghi thiết bị, kích thước ảnh, số lần đo, phần công việc được tính; không trộn thời gian đọc/vẽ ảnh vào riêng một phương pháp.

**Đầu ra:** mỗi cặp có bằng chứng khác biệt, ít nhất một phép đo, quét tham số có đường cong và giải thích nối với cơ chế.

---

## 7. Viết báo cáo trong notebook

Mỗi thí nghiệm trình bày theo thứ tự: câu hỏi → lý do chọn dữ liệu → cấu hình → hình/bảng/đường cong → giải thích → giới hạn. Đây là cách tổ chức, không phải nội dung kết luận để chép.

Theo [quy tắc báo cáo](../../agents/rules/RESULTS_REPORTING.md), mỗi con số cần ngữ cảnh:

- **What:** đại lượng, đơn vị, giá trị thực thu được.
- **Why:** câu hỏi phép đo trả lời.
- **When:** thời điểm/lần chạy tạo kết quả.
- **Where:** ảnh, biến đổi, môi trường, vị trí output trong notebook.
- **Who:** cấu hình thuật toán và nhóm thực hiện.
- **How:** tham số, định nghĩa metric, số lần chạy, cách tổng hợp.

Phần giải thích trả lời ba câu của đề: **đại lượng nào tính khác, khác ở bước nào, vì sao tạo ra hiện tượng đó**. Nêu phạm vi dữ liệu và trường hợp không thấy lợi ích; không chỉ viết “biến thể tốt hơn”. Không đưa số minh họa vào bảng kết quả thật hoặc nói đã qua kiểm chứng khi chưa chạy.

Theo [bảng điểm](../../README.md#7-thang-điểm), A 50 điểm, B 50 điểm. Tổng các hàng thí nghiệm là 34 điểm; cài đặt và kiểm chứng là 66 điểm, cần hoàn thành đủ cả hai nhóm công việc.

---

## 8. Kiểm tra cuối và nộp bài

### 8.1. Chạy sạch

1. Lưu notebook và ảnh cần dùng.
2. Restart kernel rồi Run All từ cell đầu; không chạy bù thủ công để che phụ thuộc thứ tự.
3. Chờ chạy xong; kiểm tra không traceback và đủ hình/bảng/output.
4. Lưu notebook sau khi chạy, không xóa output trước khi nộp.
5. Thử mở bản sao gói bài ở thư mục khác và chạy lại để kiểm tra đường dẫn tương đối, tài nguyên đi kèm.

### 8.2. Checklist

- [ ] Thông tin nhóm, môi trường và nguồn ảnh đầy đủ.
- [ ] Đủ 11 hàm A, 4 hàm B; giữ chữ ký theo PDF.
- [ ] Đủ năm kiểm chứng A và năm kiểm chứng B, có output.
- [ ] Có `lazy` và `frozen`, cùng bằng chứng so sánh.
- [ ] Phá hòa tái lập và một ảnh cho thấy đổi quy tắc làm đổi kết quả.
- [ ] Đủ ba cặp: bằng chứng, phép đo, đường cong quét, giải thích cơ chế.
- [ ] Prompt nguyên văn ngay trên từng cell dùng AI; không dùng AI trong cài đặt.
- [ ] Hàm góc/phân vùng dựng sẵn chỉ dùng ở phần đối chiếu được phép.
- [ ] Đã kiểm tra tên `bt1_<Nhom>.ipynb` với giảng viên nếu còn nghi vấn.
- [ ] Đúng một notebook và thư mục `images/` đủ ảnh; không phụ thuộc đường dẫn cá nhân.

README và hướng dẫn không thay notebook bắt buộc. PDF không yêu cầu nộp môi trường ảo hay toàn bộ quy tắc; theo hướng dẫn đóng gói bổ sung của giảng viên nếu có.

### 8.3. Điều kiện 0 điểm (PDF trang 9)

1. Dùng AI trong cài đặt, hoặc dùng AI ở thí nghiệm mà không hề khai báo prompt nguyên văn nào.
2. Gọi hàm phát hiện góc/phân vùng dựng sẵn bên trong hàm phải nộp.
3. Notebook không chạy được từ đầu đến cuối hoặc không có output.

---

## 9. Câu hỏi tự rà soát

| Hiện tượng | Điều cần tự kiểm tra |
| --- | --- |
| Kernel không tìm thấy thư viện | Python của kernel có trùng môi trường vừa cài? |
| Ngưỡng cho kết quả quá khác | Ảnh ở `[0,1]` hay `[0,255]`; T và ngưỡng cùng thang chưa? |
| Điểm vẽ bị đảo | Có nhầm `(row, col)` với `(x, y)`? |
| Repeatability thấp | Ánh xạ, miền hợp lệ, eps, số điểm và quy tắc ghép có nhất quán? |
| Harris–Laplace giữ gần mọi điểm | Có xét cực trị LoG theo hai tầng kề và chuẩn hóa tỉ lệ? |
| ANMS ít cải thiện độ phủ | Ứng viên có điểm ở vùng yếu không; N và lưới có giống nhau? |
| Đảo seeds ra nhiều pixel khác | Đã căn chỉnh nhãn và kiểm soát phá hòa? |
| SRG còn nhãn âm cuối cùng | Đã đạt yêu cầu phân hoạch đầy đủ? |
| Frames trông giống nhau | Có đang cùng tham chiếu tới mảng cuối? |
| Seed sensitivity khó hiểu | Định danh vùng, jitter hợp lệ và mẫu số đã rõ? |
| Chạy từng cell được, Run All lỗi | Có dùng biến lần chạy cũ hoặc đường dẫn tuyệt đối? |

Đây là câu hỏi tự đối chiếu yêu cầu. Khi cần hỗ trợ phần cài đặt, tuân thủ chính sách AI ở trang 2 thay vì gửi mã cho AI sửa.
