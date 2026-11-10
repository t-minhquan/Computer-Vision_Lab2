# Computer Vision — Bài tập 2: Đặc trưng góc và phân vùng ảnh

- **Motivation/Background**: Bài tập môn Thị giác máy tính (121036), Trường Đại học Giao thông Vận tải TP.HCM.
- **Purpose**: Tóm tắt yêu cầu và cách bắt đầu thực hiện bài tập.
- **Overview Pipeline**: Đọc đề → tự cài đặt → kiểm chứng → tự thiết kế thí nghiệm → phân tích → nộp notebook.
- **Detailed Plan**: Tổng quan; hiện trạng; quy định; hàm phải nộp; kiểm chứng; điểm số; checklist.
- **References**: [PDF gốc](pdf/Computer_Vision%2045.pdf), [hướng dẫn chi tiết](docs/shared/HUONG_DAN_THUC_HIEN.md), [quy tắc dự án](agents/README.md).
- **Created**: 2026-11-11T00:25:41+07:00
- **Last Updated**: 2026-11-11T00:34:48+07:00

---

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
- [Quy tắc dự án](agents/README.md): quy ước làm việc và tài liệu.

Hiện có PDF, bộ quy tắc và hai tài liệu này; **chưa có notebook, mã thuật toán, ảnh thí nghiệm hay kết quả đo**. Các bước hướng dẫn là công việc cần thực hiện, không phải xác nhận bài đã hoàn thành.

## 3. Bắt đầu

1. Đọc trang 2 của PDF về AI, thư viện và cách nộp.
2. Chuẩn bị môi trường theo [hướng dẫn](docs/shared/HUONG_DAN_THUC_HIEN.md#2-chuẩn-bị-môi-trường).
3. Nhóm tự viết các hàm, hoàn thành kiểm chứng trang 5–8.
4. Nhóm tự chọn dữ liệu, giả thuyết, tham số quét và phép đo cho ba cặp so sánh.
5. Restart & Run All, lưu output và đóng gói notebook cùng ảnh.

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
- [ ] Nộp **một notebook duy nhất** cùng thư mục ảnh `anh/`, dùng đường dẫn tương đối.
- [ ] Kiểm tra tên: PDF trang 2 ghi **`bt1_<Nhom>.ipynb`** dù tiêu đề là Bài tập 2; xác nhận với giảng viên trước khi đổi tên.

PDF trang 9 quy định 0 điểm nếu dùng AI trong cài đặt hoặc dùng AI ở thí nghiệm mà không hề khai báo prompt nguyên văn; gọi hàm góc/phân vùng dựng sẵn trong hàm phải nộp; hoặc notebook không chạy toàn bộ/không có output.
