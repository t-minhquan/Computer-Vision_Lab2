# Computer Vision — Bài tập 2: Đặc trưng góc và phân vùng ảnh

- **Motivation/Background**: Bài tập môn Thị giác máy tính (121036), Trường Đại học Giao thông Vận tải TP.HCM.
- **Purpose**: Tóm tắt yêu cầu và cách bắt đầu thực hiện bài tập.
- **Overview Pipeline**: Đọc đề → tự cài đặt → kiểm chứng → tự thiết kế thí nghiệm → phân tích → nộp notebook.
- **Detailed Plan**: Tổng quan; hiện trạng; quy định; hàm phải nộp; kiểm chứng; điểm số; checklist.
- **References**: [PDF gốc](pdf/Computer_Vision%2045.pdf), [hướng dẫn chi tiết](docs/shared/HUONG_DAN_THUC_HIEN.md), [quy tắc dự án](agents/README.md).
- **Created**: 2026-11-11T00:25:41+07:00
- **Last Updated**: 2026-11-11T08:40:11+07:00

---

<a id="nhom-va-phan-cong"></a>

## Nhóm và phân công

| Thành viên | MSSV | Công việc | Trạng thái |
| --- | --- | --- | --- |
| Chưa điền | Chưa điền | Chưa phân công | Chưa thực hiện |

Thông tin nhóm và phân công cho cả phần A và B được cập nhật tại đây.

Mở [hướng dẫn HTML tương tác A/B](docs/shared/PIPELINE_BAI_TAP.html) để xem sơ đồ từng bước, tìm kiếm, checklist và bản in offline.

Xem [pipeline A/B cho nhóm](docs/shared/PIPELINE_BAI_TAP.md): sơ đồ luồng, mã công việc A-01–A-06/B-01–B-06, đầu ra và checklist. Điền mã việc được giao vào bảng trên.

## 1. Tổng quan

| Phần | Thuật toán kinh điển | Biến thể | Mục tiêu đối chiếu |
| --- | --- | --- | --- |
| A1 | Harris một tỉ lệ | Harris–Laplace | Tìm lại điểm khi đổi tỉ lệ ảnh |
| A2 | Harris + chọn điểm mạnh nhất (`top_n`) | Harris + ANMS | Phân bố điểm khi giữ cùng số lượng |
| B | Region Growing | Seeded Region Growing (SRG) | Thứ tự phát triển, ngưỡng, hạt giống và phá hòa |

A1 và A2 độc lập. ANMS chạy trên ứng viên Harris **một tỉ lệ**, không phải đầu ra Harris–Laplace (PDF trang 4).

## 2. Tài liệu và hiện trạng

- [PDF đề bài, 9 trang](pdf/Computer_Vision%2045.pdf): nguồn yêu cầu chính thức.
- [Hướng dẫn thực hiện chi tiết](docs/shared/HUONG_DAN_THUC_HIEN.md): môi trường, notebook, hợp đồng hàm, kiểm chứng, thí nghiệm và đóng gói.
- [Pipeline làm việc theo phần A/B](docs/shared/PIPELINE_BAI_TAP.md): thứ tự thực hiện, phụ thuộc, đầu ra và cách phối hợp trong nhóm.
- [Quy tắc dự án](agents/README.md): quy ước làm việc và tài liệu.

Đã có PDF, bộ quy tắc, hướng dẫn, môi trường `.venv` và **hai notebook làm việc A/B**. Đã cài đặt 11 hàm phần A và 4 hàm phần B. Chưa có ảnh thí nghiệm hoặc kết quả đo; notebook chưa phải bài làm hoàn chỉnh.

## 3. Bắt đầu

1. Đọc trang 2 của PDF về AI, thư viện và cách nộp.
2. Chuẩn bị môi trường theo [hướng dẫn](docs/shared/HUONG_DAN_THUC_HIEN.md#2-chuẩn-bị-môi-trường).
3. Nhóm tự viết các hàm, hoàn thành kiểm chứng trang 5–8.
4. Nhóm tự chọn dữ liệu, giả thuyết, tham số quét và phép đo cho ba cặp so sánh.
5. Restart & Run All, lưu output và đóng gói notebook cùng ảnh.

### Cấu trúc làm việc A/B

```text
notebooks/
    phan_a_harris_va_hai_bien_the.ipynb
    phan_b_phan_vung_anh.ipynb
images/
    phan_a/
    phan_b/
experiments/
    results/
        phan_a/
        phan_b/
requirements.txt
```

| Tài nguyên | Nội dung / trạng thái |
| --- | --- |
| [Notebook phần A](notebooks/phan_a_harris_va_hai_bien_the.ipynb) | Harris, Harris–Laplace, ANMS, repeatability; kiểm chứng và hai cặp thí nghiệm. Đã cài đặt 11 hàm; chưa đo thí nghiệm. |
| [Notebook phần B](notebooks/phan_b_phan_vung_anh.ipynb) | Region Growing, SRG, frames, seed sensitivity; kiểm chứng và thí nghiệm, stale/ties. Đã cài đặt 4 hàm; chưa đo thí nghiệm. |
| [Ảnh A](images/phan_a/) / [ảnh B](images/phan_b/) | Thư mục trống để nhóm thêm ảnh tự chọn và ghi nguồn. |
| [Kết quả A](experiments/results/phan_a/) / [kết quả B](experiments/results/phan_b/) | Thư mục trống để lưu bảng số, hình và đường cong thực tế. |
| [Khai báo thư viện](requirements.txt) | Đã khóa phiên bản thư viện trực tiếp theo `.venv`, bao gồm công cụ kiểm tra notebook. |

Hai notebook chia theo phần A/B; không cần tạo thêm notebook riêng cho từng thuật toán.
Mỗi hàm có hợp đồng trong Markdown; cả hai phần đã có mã cài đặt. Mỗi kiểm chứng có
bảng kết quả và vị trí bổ sung mã; mỗi thí nghiệm có phiếu thiết kế, bảng số, chỗ vẽ và phân tích.
Cả hai notebook đã bỏ toàn bộ cell prompt; thông tin nhóm và phân công nằm trong README.

Nếu cần cài môi trường, chạy từ thư mục gốc (bước tạo template chưa chạy các lệnh cài):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter lab
```

Mở từng notebook với kernel của `.venv` và working directory là thư mục chứa notebook.
Ảnh dùng đường dẫn tương đối `../images/phan_a/` hoặc `../images/phan_b/`; bảng/hình xuất vào
[kết quả A](experiments/results/phan_a/) hoặc [kết quả B](experiments/results/phan_b/).
Chạy phần chuẩn bị trước rồi làm các mục theo thứ tự; hai notebook không import lẫn nhau.

Kiểm tra các hàm A/B từ thư mục gốc:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

Bộ kiểm tra gồm ảnh tổng hợp và trường hợp tính tay, kiểm tra seed, phá hòa, frames và sensitivity; chưa thay thế thí nghiệm trên ảnh tự chọn trong notebook.

**Trước khi nộp:** ghép nội dung A/B vào một notebook `bt1_<Nhom>.ipynb`, giữ thứ tự
định nghĩa trước khi gọi, cập nhật đường dẫn ảnh theo vị trí notebook cuối, rồi Restart & Run All
và lưu đủ output. Gói nộp cần một notebook và [thư mục ảnh](images/); chưa tạo bản tổng hợp ở bước này.

## 4. Quy định quan trọng

### Sử dụng AI

- Các cell dưới tiêu đề **“Cài đặt — KHÔNG dùng AI”** phải do sinh viên tự viết, kể cả sửa lỗi hay viết lại.
- Phần thí nghiệm được dùng AI để sinh/chỉnh sửa ảnh, dựng khung và vẽ biểu đồ. Việc chọn ảnh, tham số, phép đo và kết luận phải do nhóm quyết định.
- Mỗi cell dùng AI phải có **cell Markdown ngay phía trên**, chép nguyên văn prompt đã dùng.
- Hai tài liệu này được soạn với hỗ trợ AI để tổ chức việc thực hiện; không thay thế khai báo theo từng cell trong notebook.

### Thư viện (PDF trang 2)

| Phạm vi | Quy định |
| --- | --- |
| Luôn được phép | `numpy`, `matplotlib`, `math`, `time`; `heapq` cho SRG |
| Có điều kiện | `scipy.ndimage.gaussian_filter` để làm mờ Gauss; `cv2.imread`, `cvtColor`, `resize` để nạp/chuyển đổi ảnh; `scipy.ndimage.label` ở phần phân tích |
| Không dùng trong cài đặt | `cv2.cornerHarris`, `cv2.goodFeaturesToTrack`, `skimage.feature.corner_*`, `skimage.segmentation.*`, `cv2.watershed` |
| Đối chiếu ở thí nghiệm | Được dùng hàm dựng sẵn và được khuyến khích đối chiếu |

Không mặc định các hàm khác của SciPy/OpenCV đều được phép chỉ vì đã cài thư viện.

## 5. Các hàm phải nộp

| Nhóm | Hàm |
| --- | --- |
| Harris | `gaussian_blur`, `structure_tensor`, `harris_response`, `nms_2d`, `harris_classic` |
| Harris–Laplace | `scale_pyramid`, `log_normalized`, `harris_laplace` |
| ANMS và đường cơ sở | `anms`, `top_n` |
| Đo điểm góc | `repeatability` |
| Phân vùng | `region_growing_classic`, `srg`, `srg_step_frames`, `seed_sensitivity` |

Tổng cộng **15 hàm**. Xem chữ ký và đầu ra ở [hướng dẫn](docs/shared/HUONG_DAN_THUC_HIEN.md). PDF nhắc `grid_occupancy` là hàm cho sẵn nhưng không có mã hay chữ ký trong 9 trang; cần kiểm tra tài liệu bổ sung của giảng viên.

## 6. Kiểm chứng và thí nghiệm

**Phần A:** bàn cờ sai số ≤ 1 px; quay 30°; phóng 3×; đối chiếu OpenCV trong bán kính 2–3 px; so độ phủ lưới của `top_n` và ANMS cùng N.

**Phần B:** SRG gán nhãn mọi pixel; số nhãn bằng số hạt giống; đảo thứ tự seeds; so stack/queue/SRG bằng số pixel khác nhau; ảnh phẳng để kiểm tra phá hòa. Cần cài và so sánh cả `lazy`/`frozen`, đồng thời chứng minh đổi quy tắc phá hòa làm đổi kết quả trên ít nhất một ảnh.

Với **mỗi cặp** A1–Harris, A2–Harris, SRG–Region Growing, cần có:

1. Bằng chứng khác biệt trên dữ liệu tự chọn, kèm ít nhất một đại lượng đo.
2. Quét tham số tạo đường cong hoặc điểm chuyển pha và giải thích được cơ chế.
3. Giải thích đại lượng nào đổi, ở bước nào và vì sao gây hiện tượng quan sát.

## 7. Thang điểm

| Hạng mục | Phần A | Phần B |
| --- | ---: | ---: |
| Cài đặt kinh điển, qua kiểm chứng | 8 | 12 |
| Cài đặt biến thể, qua kiểm chứng | 11 (A1) + 11 (A2) | 18 |
| Bằng chứng khác biệt | 6 | 6 |
| Quét tham số không tầm thường | 6 | 6 |
| Giải thích thay đổi | 5 | 5 |
| **Tổng** | **50** | **50** |

Nguồn: PDF trang 9. Ba hàng thí nghiệm cộng lại là 34/100 điểm; cài đặt và kiểm chứng là 66/100 điểm.

## 8. Checklist nộp bài

- [ ] Đủ 15 hàm theo chữ ký và output kiểm chứng hai phần.
- [ ] Đủ ba cặp so sánh, mỗi cặp có bằng chứng, quét tham số, giải thích.
- [ ] Có so sánh `lazy`/`frozen` và ảnh hưởng phá hòa.
- [ ] Prompt nguyên văn ngay trên từng cell dùng AI.
- [ ] Restart & Run All thành công, lưu notebook kèm output.
- [ ] Nộp **một notebook duy nhất** cùng thư mục ảnh `images/`, dùng đường dẫn tương đối.
- [ ] Kiểm tra tên: PDF trang 2 ghi **`bt1_<Nhom>.ipynb`** dù tiêu đề là Bài tập 2; xác nhận với giảng viên trước khi đổi tên.

PDF trang 9 quy định 0 điểm nếu dùng AI trong cài đặt hoặc dùng AI ở thí nghiệm mà không hề khai báo prompt nguyên văn; gọi hàm góc/phân vùng dựng sẵn trong hàm phải nộp; hoặc notebook không chạy toàn bộ/không có output.
