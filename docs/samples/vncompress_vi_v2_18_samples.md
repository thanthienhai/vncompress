# 18 mẫu clone từ anhalu/vncompress-vi-v2


---

## 1. [compression] `comp_qa_uit_006130_2x` — doc `viquad:Louis_XVI_của_Pháp`

- split=`train` · domain=`other` · ctx_chars=17375
- requested=2.0x · target_tok=1860 · realized_tok=1768 · realized=2.10x · budget_ok=True
- extractive_ratio=0.998 · numbers_preserved=0.4868421052631579 · answerable=True · f1=0.9508
- judge_prediction: `Do không đủ quyền lực để áp đặt ý chí chính trị của mình, những cải cách đã sụp đổ trước thái độ thù địch của giới quý tộc.`
- human gold answer: `do không đủ quyền lực để áp đặt ý chí chính trị của mình, những cải cách của nhà vua đã sụp đổ trước thái độ thù địch của giới quý tộc`

**query:** Tại sao nhà vua không cải cách được triều đình và nhà nước?

**context (600 đầu):**

> Kế vị ông nội, Louis XV bị người dân căm ghét, Louis XVI rất quan tâm đến sự bất bình đang dâng cao của người dân Pháp chống lại nền quân chủ chuyên chế. Ngay từ đầu, nhà vua nỗ lực cải cách vương quốc theo các chuẩn mực của phong trào Khai sáng (chấm dứt nạn tra tấn, bãi bỏ chế độ nông nô, khoan dung đối với người Do Thái và tín hữu Kháng Cách, bỏ thuế đất đánh trên nông dân và dân thường...). Dù vậy, do không đủ quyền lực để áp đặt ý chí chính trị của mình, những cải cách của nhà vua đã sụp đổ trước thái độ thù địch của giới quý tộc. Nỗ lực hiện đại hóa vương triều nước Pháp bị thất bại.
> 
> Cá

**gold_compression (8260 chars):**

> Louis XVI nỗ lực cải cách vương quốc theo chuẩn mực phong trào Khai sáng. Dù vậy, do không đủ quyền lực để áp đặt ý chí chính trị của mình, những cải cách của nhà vua đã sụp đổ trước thái độ thù địch của giới quý tộc. Nỗ lực hiện đại hóa vương triều nước Pháp bị thất bại. Chính sự thiếu quyết đoán và quan điểm thủ cựu của Louis XVI đã khiến dân Pháp xem nhà vua như biểu tượng của sự chuyên quyền của chế độ cũ, uy tín suy giảm trầm trọng. Sự kiện nhà vua cùng hoàng tộc đào thoát đến Varennes củng cố tin đồn cho rằng nhà vua tìm kiếm sự trợ giúp từ bên ngoài nhằm đảo ngược nội tình nước Pháp. Khi lòng trung thành đối với nhà vua càng bị tổn thương nghiêm trọng thì nỗ lực lật đổ vương quyền để thiết lập nền cộng hòa càng nhận được nhiều hậu thuẫn. Không ai có thể nghi ngờ năng lực trí tuệ của Louis, nhưng rõ là nhà vua thiếu sự vững vàng và quyết đoán. Trong nhiều chiếu chỉ, nhà vua thường giải thích thiện ý của mình là chỉ nhằm mang lại ích lợi cho người dân. Khi được hỏi lý do tái triệu tập Nghị viện, Louis nói rằng, "Đây có thể là một động thái chính trị thiếu khôn ngoan, nhưng đối với ta, nó bày tỏ ước muốn được yêu thương." Quyết tâm làm một minh quân, Louis thổ lộ, "cần phải luôn hỏi ý kiến người dân; họ không bao giờ sai." Những cải cách triệt để trong lĩnh vực tài chính do Turgot và Malesherbes tiến hành khiến giới quý tộc nổi giận, chúng bị chặn lại tại Nghị viện với lập luận nhà vua không có quyền pháp lý thiết lập các loại thuế mới. Năm 1776, Turgot bị bãi chức, Males
> …[cắt]

---

## 2. [compression] `comp_qa_uit_026254_2x` — doc `viquad:Den_Haag`

- split=`train` · domain=`other` · ctx_chars=5008
- requested=2.0x · target_tok=534 · realized_tok=588 · realized=1.82x · budget_ok=True
- extractive_ratio=0.772 · numbers_preserved=0.8095238095238095 · answerable=True · f1=1.0
- judge_prediction: `Amsterdam`
- human gold answer: `Amsterdam`

**query:** Thủ đô của Hà Lan là thành phố nào?

**context (600 đầu):**

> Den Haag cũng là nơi đặt trụ sở của Chính phủ Hà Lan, nhưng lại không phải là thủ đô chính thức của nước này. Hiến pháp Hà Lan trao vai trò đó cho thành phố Amsterdam. Tuy nhiên, thành phố này là nơi có trụ sở của Eerste Kamer (Thượng nghị viện) và Tweede Kamer (Hạ nghị viện) của Nghị viện Hà Lan. Ngoài ra, Nữ hoàng Beatrix cũng sống và làm việc trong thành phố này. Các đại sứ quán ngoại quốc và phần lớn các bộ của Chính phủ đều nằm ở Den Haag, cùng với Tối cao Pháp viện Hà Lan (Hoge Raad der Nederlanden) và nhiều tổ chức hoạt động hành lang.
> 
> Hague bắt nguồn từ năm 1230, khi Count Floris IV c

**gold_compression (2892 chars):**

> Den Haag là nơi đặt trụ sở của Chính phủ Hà Lan, Nghị viện, Tối cao Pháp viện và Hội đồng Quốc gia, cùng tất cả các đại sứ quán nước ngoài, nhưng lại không phải là thủ đô theo hiến pháp (vai trò này được dành cho Amsterdam). Nữ hoàng Beatrix cũng chọn sống và làm việc tại đây.
> 
> Lịch sử của Den Haag bắt đầu từ năm 1230 khi Bá tước Floris IV mua lại một vùng đất dọc theo ao Hofvijver để xây dựng một ngôi nhà săn bắn nhỏ. Đến năm 1248, Vua William II đã quyết định mở rộng khu vực này thành một quần thể cung điện mang tên Binnenhof. Con trai ông, Bá tước Floris V, sau đó đã hoàn thành Ridderzaal (Đại sỹ đường), công trình này vẫn được sử dụng cho các sự kiện chính trị quan trọng, điển hình là bài phát biểu hàng năm từ ngai vàng. Ngay từ thế kỷ 13, The Hague đã đóng vai trò là trung tâm hành chính chính yếu của Hà Lan.
> 
> Trong suốt Chiến tranh Tám mươi năm, việc không có tư cách thành phố chính thức khiến quân đội Tây Ban Nha dễ dàng chiếm đóng khu vực này. Năm 1575, hội đồng từng xem xét phá hủy hoàn toàn thành phố nhưng đã bị William of Orange can thiệp kịp thời. Từ năm 1588, The Hague chính thức trở thành ghế của chính phủ Cộng hòa Hà Lan. Nhằm mục đích kiểm soát quyền lực, The Hague đã không được cấp tư cách thành phố dù sở hữu rất nhiều đặc quyền.
> 
> Mãi đến năm 1806, vua Louis Bonaparte mới chính thức cấp tư cách thành phố cho The Hague. Sau khi Chiến tranh Napoleon kết thúc, Brussels và Amsterdam luân phiên đảm nhận vai trò thủ đô, trong khi chính phủ vẫn luôn đặt tại The Hagu
> …[cắt]

---

## 3. [compression] `comp_qa_uit_002147_2x` — doc `viquad:Thần_học`

- split=`train` · domain=`other` · ctx_chars=6206
- requested=2.0x · target_tok=661 · realized_tok=589 · realized=2.25x · budget_ok=True
- extractive_ratio=0.893 · numbers_preserved=0.0 · answerable=True · f1=1.0
- judge_prediction: `Theologia`
- human gold answer: `Theologia`

**query:** Thần học trong tiếng La Tinh là gì?

**context (600 đầu):**

> Thần học là ngành nghiên cứu về các thần thánh, hay rộng hơn là về niềm tin tôn giáo, thực hành và trải nghiệm tôn giáo, về linh hồn. Thần học giúp các nhà thần học hiểu rõ hơn về truyền thống tôn giáo của chính mình, về các truyền thống tôn giáo khác, cũng như so sánh giữa các truyền thống tôn giáo khác nhau, … Thần học, nguyên nghĩa trong tiếng La Tinh là Theologia, ghép 2 từ trong tiếng Hy Lạp là Theos (nghĩa là thần linh) và logos (nghĩa là lời); vậy Theologia là môn học nghiên cứu về những lời, lý lẽ phù hợp với Thiên Chúa. Ngày nay, thần học được giảng dạy trong các chủng viện, trường dò

**gold_compression (2942 chars):**

> Thần học là ngành nghiên cứu về thần thánh, niềm tin, thực hành và trải nghiệm tôn giáo, linh hồn. Từ "Theologia" (tiếng La Tinh) ghép từ "Theos" (thần linh) và "logos" (lời), nghĩa là môn học về lý lẽ phù hợp Thiên Chúa. Ngày nay, thần học được giảng dạy tại chủng viện, trường dòng, học viện, đại học.
> 
> Có nhiều định nghĩa: Thánh Augustine cho thần học là lập luận về Thiên Chúa; Richard Hooker coi đó là khoa học về điều thiêng liêng. Thần học bắt đầu từ giả thuyết sự thiêng liêng hiện hữu dạng vật lý, siêu nhiên, tâm lý hay thực thể xã hội, qua trải nghiệm tôn giáo, tâm linh hoặc ghi chép lịch sử. Nghiên cứu giả thuyết này thuộc triết học tôn giáo, tâm lý tôn giáo, thần kinh học. Dưới tư cách khoa học, thần học thiết lập, đào sâu kinh nghiệm, khái niệm để tìm bằng chứng về hữu thể, khám phá điều huyền bí.
> 
> Bản chất thần học là tìm hiểu, suy tư siêu nhiên, thiêng liêng qua đức tin. Mọi thắc mắc về thần linh, ma quỷ, thiên đàng, hỏa ngục đều liên quan. Nghiên cứu thần học giúp thần học gia hiểu truyền thống tôn giáo, thần thoại tộc người, khám phá bản chất thần học. Thần học lưu truyền, biện minh, so sánh, thách thức truyền thống tôn giáo, chuyển tải tình huống hiện thời, lý giải thế giới.
> 
> Triết học Ấn Độ giáo có truyền thống suy đoán về vũ trụ, Thượng đế (Brahman, Paramatma, Bhagavan), Atman (linh hồn). Thuật ngữ Sankrit "Darshana" nghĩa là quan điểm. Thần học Vaishnava nghiên cứu sự biểu lộ của hàng ngàn vị thần. Gần đây, lĩnh vực này được tổ chức hàn lâm châu Âu đảm nhận, v
> …[cắt]

---

## 4. [compression] `comp_qa_uit_014368_4x` — doc `viquad:Lý_thuyết_tập_hợp`

- split=`train` · domain=`other` · ctx_chars=5361
- requested=4.0x · target_tok=289 · realized_tok=294 · realized=3.94x · budget_ok=True
- extractive_ratio=1.000 · numbers_preserved=0.0 · answerable=True · f1=1.0
- judge_prediction: `Hội và giao.`
- human gold answer: `hội và giao`

**query:** Tập hợp có những phép toán cơ bản nào?

**context (600 đầu):**

> Ngôn ngữ của lý thuyết tập hợp được dùng trong định nghĩa của gần như tất cả các đối tượng toán học, như hàm số, và các khái niệm lý thuyết tập hợp được đưa nhiều chương trình giảng dạy toán học. Các sự kiện cơ bản về tập hợp và phần tử trong tập hợp có thể được mang ra giới thiệu ở cấp tiểu học, cùng với sơ đồ Venn, để học về tập hợp các đối tượng vật lý thường gặp. Các phép toán cơ bản như hội và giao có thể được học trong bối cảnh này. Các khái niệm cao hơn như bản số là phần tiêu chuẩn của chương trình toán học của sinh viên đại học.
> 
> Cantor phân loại các tập hợp, đặc biệt là những tập hợp

**gold_compression (1295 chars):**

> Ngôn ngữ của lý thuyết tập hợp được dùng trong định nghĩa của gần như tất cả các đối tượng toán học, như hàm số, và các khái niệm lý thuyết tập hợp được đưa nhiều chương trình giảng dạy toán học. Các sự kiện cơ bản về tập hợp và phần tử trong tập hợp có thể được mang ra giới thiệu ở cấp tiểu học, cùng với sơ đồ Venn, để học về tập hợp các đối tượng vật lý thường gặp. Các phép toán cơ bản như hội và giao có thể được học trong bối cảnh này. Các khái niệm cao hơn như bản số là phần tiêu chuẩn của chương trình toán học của sinh viên đại học. Cantor phân loại các tập hợp, đặc biệt là những tập hợp vô hạn, theo Lực lượng của chúng. Đối với tập hợp hữu hạn, đây là số lượng các phần tử của chúng. Ông gọi hai tập hợp " có lực lượng bằng nhau" khi chúng được ánh xạ song ánh với nhau, tức là khi có một mối quan hệ một-một giữa các phần tử của chúng. Cái được định nghĩa là sự đồng nhất lực lượng là một quan hệ tương đương, và một lực lượng hay số phần tử của một tập hợp M theo Cantor, là lớp tương đương của các tập hợp có lực lượng bằng M. Ông là người đầu tiên quan sát thấy rằng có những lực lựong vô hạn khác nhau. Tập hợp các số tự nhiên, và tất cả các tập hợp có lực lượng bằng nó, được Cantor gọi là 'Tập hợp đếm được, tất cả các tập hợp vô hạn khác được gọi là tập hợp không đếm được.

---

## 5. [compression] `comp_qa_uit_026635_4x` — doc `viquad:Người`

- split=`train` · domain=`other` · ctx_chars=18791
- requested=4.0x · target_tok=1043 · realized_tok=1125 · realized=3.71x · budget_ok=True
- extractive_ratio=0.997 · numbers_preserved=0.19318181818181818 · answerable=True · f1=0.7368
- judge_prediction: `Khả năng thực hiện các suy luận trừu tượng, ngôn ngữ, xem xét nội tâm và dùng nhiều công cụ hơn các loài khác.`
- human gold answer: `thực hiện các suy luận trừu tượng, ngôn ngữ và xem xét nội tâm`

**query:** Nhờ vào bộ não phát triển vượt bậc mà loài người có được điều gì?

**context (600 đầu):**

> Loài người (theo phân loại học là Homo sapiens, tiếng La-tinh nghĩa là "người thông thái" hay "người thông minh", nên cũng được dịch sang tiếng Việt là trí nhân hay người tinh khôn) là loài duy nhất còn sống của tông Hominini, thuộc lớp động vật có vú. Con người là một loài sinh vật có bộ não tiến hóa rất cao cho phép thực hiện các suy luận trừu tượng, ngôn ngữ và xem xét nội tâm. Điều trên kết hợp với một cơ thể đứng thẳng cho phép giải phóng hai chi trước khỏi việc di chuyển và được dùng vào việc cầm nắm, cho phép con người dùng nhiều công cụ hơn tất cả những loài khác.
> 
> Cũng như những loài 

**gold_compression (5078 chars):**

> Loài người (Homo sapiens) là loài sinh vật có bộ não tiến hóa rất cao cho phép thực hiện các suy luận trừu tượng, ngôn ngữ và xem xét nội tâm. Điều trên kết hợp với một cơ thể đứng thẳng cho phép giải phóng hai chi trước khỏi việc di chuyển và được dùng vào việc cầm nắm, cho phép con người dùng nhiều công cụ hơn tất cả những loài khác. Con người là một sinh vật xã hội, sống bầy đàn, có sự phân thứ bậc nhất định xác định từ cọ xát và truyền thống. Hơn thế nữa, con người cũng rất thành thạo việc sử dụng ngôn ngữ trong giao tiếp, để biểu lộ những ý kiến riêng của mình và trao đổi thông tin. Con người tạo ra những xã hội phức tạp trong đó có những nhóm hỗ trợ nhau và đối nghịch nhau ở từng mức độ, có thể từ những cá nhân trong gia đình cho đến những quốc gia rộng lớn. Giao tiếp xã hội giữa con người và con người đã góp phần tạo nên những truyền thống, nghi thức, quy tắc đạo đức, giá trị, chuẩn mực xã hội, và cả luật pháp. Tất cả cùng nhau tạo nên những nền tảng của xã hội loài người. Con người cũng rất chú ý đến cái đẹp và thẩm mỹ, cùng với nhu cầu muốn bày tỏ mình, đã tạo nên những sự đổi với về văn hóa như nghệ thuật, văn chương và âm nhạc. Sự tiến hóa của con người được đánh dầu bằng những dấu hiệu sinh học khác nhau, bao gồm sự phát triển của hộp sọ và cả bộ não lên đến mức 1.400 cm³; về thể tích, cao hơn gấp đôi tinh tinh hay khỉ đột. Những phần của bộ não con người cũng phát triển khác so với các loài linh trưởng khác cho phép xuất hiện thêm phần ngôn ngữ. Những nhà khoa họ
> …[cắt]

---

## 6. [compression] `comp_qa_uit_001037_4x` — doc `viquad:Albert_Einstein`

- split=`train` · domain=`other` · ctx_chars=23435
- requested=4.0x · target_tok=1251 · realized_tok=1188 · realized=4.21x · budget_ok=True
- extractive_ratio=0.808 · numbers_preserved=0.030303030303030304 · answerable=True · f1=0.7273
- judge_prediction: `Công trình về hiệu ứng quang điện.`
- human gold answer: `hiệu ứng quang điện`

**query:** Công trinh nào của ông có tính chất khai sinh ra thuyết lượng tử?

**context (600 đầu):**

> Albert Einstein (tiếng Đức: [ˈalbɐt ˈaɪnʃtaɪn] ( nghe), phiên âm: Anh-xtanh; 14 tháng 3 năm 1879 – 18 tháng 4 năm 1955) là nhà vật lý lý thuyết người Đức, người đã phát triển thuyết tương đối tổng quát, một trong hai trụ cột của vật lý hiện đại (trụ cột kia là cơ học lượng tử). Mặc dù được biết đến nhiều nhất qua phương trình về sự tương đương khối lượng-năng lượng E = mc2 (được xem là "phương trình nổi tiếng nhất thế giới"), ông lại được trao Giải Nobel Vật lý năm 1921 "cho những cống hiến của ông đối với vật lý lý thuyết, và đặc biệt cho sự khám phá ra định luật của hiệu ứng quang điện". Côn

**gold_compression (5495 chars):**

> Công trình về hiệu ứng quang điện của Albert Einstein, được công bố trong bài báo khoa học vào năm 1905, mang tính chất bước ngoặt lịch sử và triệt để, khai sinh ra và củng cố nền tảng vững chắc cho lý thuyết lượng tử hiện đại, đồng thời lật đổ hoàn toàn nền tảng của vật lý cổ điển. Trước công trình mang tính đột phá này, hiện tượng quang điện – tức sự phát xạ của các electron khi một bức xạ điện từ, cụ thể là ánh sáng, chiếu lên một bề mặt vật chất – đã được quan sát thực nghiệm nhưng hoàn toàn không thể giải thích được thông qua lăng kính của vật lý cổ điển. Theo thuyết sóng ánh sáng cổ điển do James Clerk Maxwell xây dựng, năng lượng của sóng điện từ được phân bố liên tục và đồng đều trên toàn bộ mặt sóng. Do đó, năng lượng hấp thụ bởi các electron trong kim loại phải tích lũy một cách từ từ. Lý thuyết cổ điển dự đoán rằng, dù tần số của ánh sáng chiếu tới có thấp đến đâu, chỉ cần chiếu sáng đủ lâu và với cường độ đủ lớn, electron cuối cùng cũng sẽ tích lũy đủ năng lượng để thoát khỏi bề mặt kim loại. Hơn nữa, theo vật lý cổ điển, động năng ban đầu của các electron bị bắn ra phải tỷ lệ thuận với cường độ của chùm ánh sáng chiếu tới. Tuy nhiên, các kết quả thực nghiệm lại chỉ ra một bức tranh hoàn toàn khác biệt và kỳ lạ. Thứ nhất, không có độ trễ thời gian nào được ghi nhận; electron phát xạ ngay lập tức ngay khi ánh sáng vừa chiếu tới, ngay cả khi cường độ ánh sáng cực kỳ yếu. Thứ hai, động năng của electron không phụ thuộc vào cường độ ánh sáng, mà lại phụ thuộc trực tiế
> …[cắt]

---

## 7. [compression] `comp_qa_uit_001457_8x` — doc `viquad:Cộng_hòa_Dân_chủ_Đức`

- split=`train` · domain=`other` · ctx_chars=20207
- requested=8.0x · target_tok=537 · realized_tok=523 · realized=8.22x · budget_ok=True
- extractive_ratio=0.803 · numbers_preserved=0.056 · answerable=True · f1=1.0
- judge_prediction: `Hoa Kỳ, Anh, Pháp và Liên Xô.`
- human gold answer: `Hoa Kỳ, Anh, Pháp và Liên Xô`

**query:** Hiệp định Potsdam là hiệp định giữa các nước nào?

**context (600 đầu):**

> Cộng hòa Dân chủ Đức (tiếng Đức: Deutsche Demokratische Republik, DDR; thường được gọi là Đông Đức) là một quốc gia nay không còn nữa, tồn tại từ 1949 đến 1990 theo định hướng xã hội chủ nghĩa tại phần phía đông nước Đức ngày nay. Cộng hòa Dân chủ Đức được thành lập tại vùng quản lý của Quân đội Xô-viết tại Đức ngày 7 tháng 10 năm 1949, sau khi Nhà nước Cộng hòa Liên bang Đức (Tây Đức) ra đời tại phần phía tây trong khu vực do Pháp, Anh và Hoa Kỳ quản lý. Đông Berlin là thủ đô của Cộng hòa Dân chủ Đức.
> 
> Thành lập từ năm 1949, tuy nhiên, đến năm 1955, Đông Đức mới tuyên bố đầy đủ quyền tự trị n

**gold_compression (2429 chars):**

> Hiệp định Potsdam, hay còn được gọi là Tuyên bố Potsdam, là một thỏa thuận quốc tế mang tính lịch sử được ký kết vào cuối giai đoạn Chiến tranh thế giới thứ hai. Hội nghị diễn ra tại thành phố Potsdam, nằm gần thủ đô Berlin của nước Đức, từ ngày 17 tháng 7 đến ngày 2 tháng 8 năm 1945. Mục đích chính của hội nghị này là bàn bạc và đưa ra các quyết định quan trọng về việc quản lý nước Đức sau khi phát xít Đức đầu hàng vô điều kiện, cũng như thiết lập một trật tự thế giới mới sau chiến tranh. 
> 
> Bốn cường quốc chiến thắng tham gia ký kết hiệp định này bao gồm Hoa Kỳ, Anh, Pháp và Liên Xô. Cần lưu ý một chi tiết lịch sử quan trọng là trong giai đoạn đầu của hội nghị, cường quốc tham gia thay mặt cho Anh Quốc là Thủ tướng Winston Churchill. Tuy nhiên, do kết quả của cuộc bầu cử tổng thống tại Anh, Churchill đã phải từ chức và được thay thế bằng Thủ tướng mới Clement Attlee trong nửa sau của hội nghị. Đại diện cho Hoa Kỳ là Tổng thống Harry S. Truman, người vừa mới lên thay thế cố Tổng thống Franklin D. Roosevelt. Đại diện cho Liên Xô là lãnh tụ Joseph Stalin. Đại diện cho Pháp là Tổng thống chính phủ lâm thời Charles de Gaulle. 
> 
> Bốn cường quốc này đã cùng nhau thảo luận và đưa ra nhiều quyết định liên quan đến việc phân chia lãnh thổ Đức. Theo đó, nước Đức cùng với thủ đô Berlin sẽ bị chia cắt thành bốn khu vực chiếm đóng quân sự, mỗi khu vực do một trong bốn cường quốc quản lý. Ngoài ra, hội nghị cũng đưa ra các quyết định về việc phi quân sự hóa, phi phát xít hóa, phi độc quyền 
> …[cắt]

---

## 8. [compression] `comp_qa_uit_017120_8x` — doc `viquad:Trung_Quốc_Quốc_dân_Đảng`

- split=`train` · domain=`other` · ctx_chars=23539
- requested=8.0x · target_tok=635 · realized_tok=626 · realized=8.13x · budget_ok=True
- extractive_ratio=0.994 · numbers_preserved=0.3673469387755102 · answerable=True · f1=1.0
- judge_prediction: `Ngày 10 tháng 10 năm 1919.`
- human gold answer: `ngày 10 tháng 10 năm 1919`

**query:** Tôn Trung Sơn đã đổi tên Đảng khi nào?

**context (600 đầu):**

> Trung Quốc Quốc dân Đảng (tiếng Trung: 中國國民黨, gọi tắt là Quốc dân Đảng, tên tiếng Anh là "Kuomintang of China" hay "Chinese Nationalist Party", viết tắt "KMT") do Tôn Trung Sơn và các đồng chí của ông sáng lập và tồn tại cho đến nay, cũng là một trong số các chính đảng sớm nhất tại châu Á. Tiền thân của chính đảng này là đoàn thể cách mạng Hưng Trung hội thành lập tại Hawaii vào năm 1894, sau đó lần lượt cải tổ thành Trung Quốc Đồng minh hội, Quốc dân Đảng và Trung Hoa Cách mệnh Đảng, đến ngày 10 tháng 10 năm 1919 sau khi Tôn Trung Sơn cải tổ thì đổi sang danh xưng hiện tại.
> 
> Trung Quốc Quốc d

**gold_compression (2860 chars):**

> Trung Quốc Quốc dân Đảng do Tôn Trung Sơn và các đồng chí sáng lập. Tiền thân là Hưng Trung hội thành lập tại Hawaii năm 1894, sau cải tổ thành Trung Quốc Đồng minh hội, Quốc dân Đảng và Trung Hoa Cách mệnh Đảng. Ngày 10 tháng 10 năm 1919, sau khi Tôn Trung Sơn cải tổ, đảng đổi sang danh xưng hiện tại.
> 
> Năm 1905, Tôn Trung Sơn liên hiệp lực lượng với các đoàn thể chống phong kiến Hoa Hưng hội, Quang Phục hội tại Tokyo, Nhật Bản để hình thành Trung Quốc Đồng minh Hội vào ngày 20 tháng 8 năm 1905. Tổ chức lên kế hoạch và ủng hộ Cách mạng Tân Hợi năm 1911 và thành lập Trung Hoa Dân Quốc vào ngày 1 tháng 1 năm 1912. Ngày 11 tháng 8 năm 1912, Đồng minh Hội, Thống nhất Cộng hòa Đảng, Quốc dân Công Đảng, Quốc dân Cộng tiến Đảng và Cộng hòa Thực tiến hội hợp nhất, đến ngày 13 tháng 8 tuyên bố: "chế độ cộng hòa, quốc dân là chủ thể của đất nước, chúng tôi vì để mọi người không quên lẽ đó, nên để tên là Quốc dân Đảng". Ngày 25 tháng 8 năm 1912, Quốc dân Đảng được thành lập tại Hội quán Hồ Quảng tại Bắc Kinh, chính thức tuyên bố Quốc dân Đảng ra đời.
> 
> Thành viên có thế lực nhất của đảng là Tống Giáo Nhân, ông huy động ủng hộ lớn từ giới thân sĩ và thương nhân cho Quốc dân Đảng nhằm chủ trương một chế độ dân chủ nghị viện lập hiến. Đảng phản đối phái chủ nghĩa quân chủ lập hiến và tìm cách kiềm chế quyền lực của Viên Thế Khải. Quốc dân Đảng giành được số ghế nhiều nhất trong bầu cử quốc hội lần thứ nhất vào tháng 12 năm 1912-tháng 1 năm 1913. Tuy nhiên, Viên Thế Khải nhanh chóng bắt đầu 
> …[cắt]

---

## 9. [compression] `comp_qa_uit_001795_8x` — doc `viquad:François_Mitterrand`

- split=`train` · domain=`other` · ctx_chars=23531
- requested=8.0x · target_tok=604 · realized_tok=605 · realized=7.99x · budget_ok=True
- extractive_ratio=0.803 · numbers_preserved=0.07482993197278912 · answerable=True · f1=0.6087
- judge_prediction: `Giáo sư luật học quốc tế người Pháp, được chỉ định làm cố vấn pháp lý cho Hoàng đế Haile Selassie của Ethiopia.`
- human gold answer: `giáo sư luật Gaston Jèze, người từng được chỉ định làm cố vấn pháp lý của Negus tại Ethiopia tháng 1 năm 1936`

**query:** Gaston Jèze là ai?

**context (600 đầu):**

> Mitterrand sinh tại Jarnac, Charente, và được rửa tội với tên gọi François Maurice Adrien Marie Mitterrand. Gia đình ông nhiệt tâm theo Cơ đốc giáo La Mã và rất bảo thủ. Cha ông, Joseph Gilbert Félix, làm kỹ sư cho la Compagnie Paris Orléans, cha dượng của ông là một người làm dấm và sau này làm chủ tịch liên đoàn những người làm dấm (Fédération des syndicats de fabricants de vinaigre). Bà ngoại của Joseph là một phụ nữ quý tộc, hậu duệ của cả Fernando III của Castile và Jean de Brienne của Jerusalem. Mẹ Mitterrand là Marie Gabrielle Yvonne Lorrain, một cháu họ xa của Giáo hoàng John XXII. Ông

**gold_compression (2732 chars):**

> Ông đã tham gia một cách tích cực và sâu sắc vào các cuộc tuần hành bài ngoại mang tính chất bạo lực và quyết liệt chống lại cái được gọi là "sự xâm lăng métèque" diễn ra vào tháng 2 năm 1935. Đây là một loạt các cuộc biểu tình lớn do các sinh viên theo chủ nghĩa dân tộc cực hữu tổ chức tại thủ đô Paris, nhằm vào những người nước ngoài mà họ cho là đang đe dọa trực tiếp đến sinh kế và vị thế của người lao động bản địa Pháp. Trong những cuộc tuần hành này, ông cùng với những người biểu tình khác đã giương cao các biểu ngữ, hô to các khẩu hiệu đậm chất bài ngoại, và nhiều lần đụng độ gay gắt với lực lượng cảnh sát được điều động đến để duy trì trật tự công cộng. Sự kiện này đánh dấu một bước ngoặt quan trọng, cho thấy sự cam kết mạnh mẽ và không thể chối cãi của ông đối với các phong trào dân tộc chủ nghĩa cấp tiến đang trỗi dậy mạnh mẽ tại nước Pháp trong giai đoạn giữa thập niên 1930.
> 
> Không dừng lại ở đó, chỉ vài tháng sau, ông lại tiếp tục lăn mình vào những cuộc tuần hành phản đối dữ dội khác nhắm vào giáo sư luật Gaston Jèze. Vào tháng 1 năm 1936, giáo sư Jèze, một học giả luật học quốc tế người Pháp, đã được chính thức chỉ định làm cố vấn pháp lý cho Negus, tức Hoàng đế Haile Selassie của Ethiopia. Ethiopia lúc bấy giờ đang trong giai đoạn vô cùng khó khăn khi phải đối mặt với cuộc xâm lược tàn bạo của phát xít Ý. Việc một học giả người Pháp nhận lời cố vấn cho chính phủ Ethiopia đã châm ngòi cho sự phẫn nộ tột độ của giới sinh viên và thanh niên theo chủ nghĩa dân tộc c
> …[cắt]

---

## 10. [compression(val)] `comp_qa_uit_018371_2x` — doc `viquad:Nhà_Minh`

- split=`validation` · domain=`other` · ctx_chars=23666
- requested=2.0x · target_tok=2529 · realized_tok=2652 · realized=1.91x · budget_ok=True
- extractive_ratio=1.000 · numbers_preserved=0.6 · answerable=True · f1=0.9167
- judge_prediction: `Sau khi Minh Thành Tổ Chu Đệ dời đô đến phủ Thuận Thiên.`
- human gold answer: `Minh Thành Tổ Chu Đệ dời đô đến phủ Thuận Thiên`

**query:** Sau khi sự kiện nào xảy ra thì phủ Ứng Thiên được đổi tên thành Nam Kinh?

**context (600 đầu):**

> Nhà Minh (chữ Hán: 明朝, Hán Việt: Minh triều; 23 tháng 1 năm 1368 - 25 tháng 4 năm 1644 [chú thích 1]) là triều đại cuối cùng do người Hán kiến lập trong lịch sử Trung Quốc. Năm 1368, Chu Nguyên Chương sau khi tiêu diệt các thế lực quần hùng, tại phủ Ứng Thiên đăng cơ, quốc hiệu Đại Minh, do hoàng thất họ Chu, nên còn được gọi là Chu Minh. Đầu thời kiến quốc, Minh Thái Tổ Chu Nguyên Chương định đô tại phủ Ứng Thiên (nay là Nam Kinh), đến năm Vĩnh Lạc thứ 19 (1421), Minh Thành Tổ Chu Đệ dời đô đến phủ Thuận Thiên (nay là Bắc Kinh), kinh sư cũ đổi thành Nam Kinh.
> 
> Thời kỳ đầu triều Minh, qua chín

**gold_compression (12348 chars):**

> Năm 1368, Chu Nguyên Chương sau khi tiêu diệt các thế lực quần hùng, tại phủ Ứng Thiên đăng cơ, quốc hiệu Đại Minh, do hoàng thất họ Chu, nên còn được gọi là Chu Minh. Đầu thời kiến quốc, Minh Thái Tổ Chu Nguyên Chương định đô tại phủ Ứng Thiên (nay là Nam Kinh), đến năm Vĩnh Lạc thứ 19 (1421), Minh Thành Tổ Chu Đệ dời đô đến phủ Thuận Thiên (nay là Bắc Kinh), kinh sư cũ đổi thành Nam Kinh. Thời kỳ đầu triều Minh, qua chính sách nghỉ ngơi lại sức của Chu Nguyên Chương, quốc lực triều Minh khôi phục nhanh chóng, sử xưng Hồng Vũ chi trị. Đến thời kỳ Minh Thành Tổ Chu Đệ, quốc thế đạt đỉnh, những năm Vĩnh Lạc khoa trương lãnh thổ, còn phái khiển Trịnh Hòa bảy lần hạ Tây Dương, học giả hiện đại gọi là Vĩnh Lạc thịnh thế. Sau đó, thời kỳ Nhân Tông và Tuyên Tông cũng là thời hưng thịnh, sử xưng Nhân Tuyên chi trị. Tuy nhiên, thời kỳ Anh Tông và Cảnh Thái Đế, trải qua sự biến Thổ Mộc bảo, quốc lực trung suy. Sau khi Thế Tông đăng cơ, phát sinh tranh chấp Đại lễ nghị, sau khi thanh trừ thế lực hoạn quan và quyền thần hoàng đế tổng quản triều cương, thực hành Gia Tĩnh tân chính, song sau này không quan tâm triều chính. Sau khi Minh Thế Tông từ trần, trải qua Long Khánh tân chính và Vạn Lịch trung hưng, quốc lực được khôi phục. Trung kỳ thời Thần Tông, hoàng đế dần lơ đãng triều chính, sử xưng Vạn Lịch đãi chính, bắt đầu chính trị hỗn loạn thời vãn Minh. Thời Hi Tông, đám hoạn quan làm loạn triều cương, sau khi Tư Tông kế vị thì bị diệt trừ. Tuy nhiên, do Tư Tông có quyết sách sai lầm,
> …[cắt]

---

## 11. [compression(val)] `comp_qa_uit_025634_8x` — doc `viquad:Kinh_tế_Hàn_Quốc`

- split=`validation` · domain=`other` · ctx_chars=5728
- requested=8.0x · target_tok=155 · realized_tok=173 · realized=7.17x · budget_ok=True
- extractive_ratio=0.977 · numbers_preserved=0.4782608695652174 · answerable=True · f1=1.0
- judge_prediction: `25.000 USD`
- human gold answer: `25.000 USD`

**query:** Mức GDP bình quân đầu người của Hàn Quốc vào năm 2007 là bao nhiêu?

**context (600 đầu):**

> Kinh tế Hàn Quốc là nền kinh tế phát triển, đứng thứ tư ở châu Á và đứng thứ 11 trên thế giới theo GDP năm 2016. Sau Chiến tranh Triều Tiên, kinh tế Hàn Quốc đã phát triển nhanh chóng, từ một trong những nước nghèo nhất thế giới trở thành một trong những nước phát triển nhất. Cuối thế kỷ 20, Hàn Quốc là một trong những nước có tốc độ tăng trưởng kinh tế nhanh nhất trong lịch sử thế giới hiện đại. GDP (PPP) bình quân đầu người của đất nước đã nhảy vọt từ 100 USD vào năm 1963 lên mức 10.000 USD vào năm 1995 và 25.000 USD vào năm 2007. Bất chấp các ảnh hưởng nặng nề từ cuộc khủng hoảng kinh tế ch

**gold_compression (776 chars):**

> GDP (PPP) bình quân đầu người của Hàn Quốc đã nhảy vọt từ 100 USD vào năm 1963 lên mức 10.000 USD vào năm 1995 và 25.000 USD vào năm 2007. Năm 2007, Goldman Sachs phân tích rằng nếu duy trì tốc độ tăng trưởng GDP bình quân 5% mỗi năm, Hàn Quốc sẽ trở thành nền kinh tế lớn thứ 9 thế giới vào năm 2025 với GDP bình quân đầu người là 52.000 USD, và 25 năm sau nữa sẽ vượt qua tất cả các nước ngoại trừ Hoa Kỳ để trở thành nước có GDP đầu người thứ hai trên thế giới, với bình quân đầu người là 81.000 USD. Khi Tướng Park Chung-hee nắm quyền vào năm 1961, Hàn Quốc đã có một thu nhập bình quân đầu người ít hơn 80 USD mỗi năm. Trong thời gian đó, Hàn Quốc chủ yếu phụ thuộc vào viện trợ nước ngoài, chủ yếu là từ Mỹ để đổi lấy sự tham gia của Hàn Quốc trong chiến tranh Việt Nam.

---

## 12. [extras(unanswerable)] `comp_qa_uit_002026_4x` — doc `viquad:Lưu_vực`

- split=`train` · domain=`other` · ctx_chars=4487
- requested=4.0x · target_tok=249 · realized_tok=238 · realized=4.19x · budget_ok=True
- extractive_ratio=0.765 · numbers_preserved=0.3333333333333333 · answerable=False · f1=0.3636
- judge_prediction: `Biển Caspian`
- human gold answer: `biển Caspian, biển Aral và nhiều hồ nhỏ hơn`

**query:** Lòng chảo nội lục của châu Á đổ vào đâu nhiều nhất?

**context (600 đầu):**

> Lưu vực là phần diện tích bề mặt đất trong tự nhiên mà mọi lượng nước mưa khi rơi xuống sẽ tập trung lại và thoát vào một lối thoát thông thường, chẳng hạn như vào sông, vịnh hoặc các phần nước khác. Các lưu vực thoát nước bao gồm tất cả các nước bề mặt từ dòng chảy mưa, tuyết, và các dòng suối gần đó chạy theo hướng dốc về phía lối thoát chung, cũng như nước ngầm dưới bề mặt trái đất. Các lưu vực thoát nước kết nối với các lưu vực thoát nước khác ở độ cao thấp theo mô hình phân cấp, với các bể chứa nhỏ hơn, và lần lượt đổ vào một khe thông thường khác.
> 
> Lòng chảo nội lục là nội lục lưu vực mà

**gold_compression (1083 chars):**

> Lòng chảo nội lục là một vùng lưu vực kín, nơi nước từ mưa, sông suối chảy vào nhưng không có đường thoát ra đại dương. Khoảng 18% tổng diện tích đất liền trên Trái Đất thuộc các lòng chảo nội lục, nước tập trung tại các hồ, biển nội địa hoặc bồn trũng địa phương rồi bay hơi hoặc ngấm xuống đất.
> 
> Phần lớn diện tích nội lục lớn nhất nằm ở châu Á, bao gồm các lưu vực đổ nước vào biển Caspian, biển Aral và nhiều hồ nhỏ khác. Ở Bắc Mỹ, Đại Bồn Địa (Hoa Kỳ) là một tập hợp các lưu vực kín liền kề nhau, nổi bật với Hồ Great Salt. Tại châu Phi, phần lớn sa mạc Sahara, lưu vực sông Okavango (vùng Kalahari) và vùng cao nguyên quanh Hồ Lớn cũng không có lối thoát ra biển. Châu Úc có nhiều lòng chảo nội lục khô cằn, trong khi Bán đảo Ả Rập cũng chứa các vùng nội lục rộng lớn. Ngoài ra, một số khu vực ở México và dãy núi Andes (Nam Mỹ) cũng hình thành các lưu vực kín do địa hình núi cao chắn đường chảy tràn.
> 
> Đặc điểm chung của các lòng chảo này là sự tích tụ khoáng chất, tạo ra các hồ mặn hoặc vùng đất ngập nước đặc thù, đóng vai trò sinh thái quan trọng tại các khu vực khô hạn.

---

## 13. [extras(unanswerable)] `comp_qa_uit_022584_4x` — doc `viquad:Cuba`

- split=`train` · domain=`other` · ctx_chars=23485
- requested=4.0x · target_tok=1270 · realized_tok=1129 · realized=4.50x · budget_ok=True
- extractive_ratio=0.958 · numbers_preserved=0.14678899082568808 · answerable=False · f1=0.0
- judge_prediction: `Quần đảo Đại Antilles.`
- human gold answer: `biển Caribe`

**query:** Hòn đảo lớn nhất của Cộng hòa Cuba có vị trí tại đâu?

**context (600 đầu):**

> Cuba, tên gọi chính thức là Cộng hòa Cuba (tiếng Tây Ban Nha: Cuba hay República de Cuba, IPA: [re'puβlika ðe 'kuβa]) là Quốc gia bao gồm đảo Cuba (hòn đảo hình con cá sấu vươn dài trên biển Caribe, cũng là hòn đảo lớn nhất của quần đảo Đại Antilles), cùng với đảo Thanh Niên (Isla de la Juventud) và các đảo nhỏ xung quanh. Cuba nằm ở phía bắc Vùng Caribe ở giao điểm của ba miền biển lớn: Biển Caribe, Vịnh México và Đại Tây Dương. Cuba nằm ở phía nam miền đông Hoa Kỳ và Bahamas, phía tây Quần đảo Turks và Caicos và Haiti và phía đông México. Quần đảo Cayman và Jamaica ở phía nam.
> 
> Theo văn tịch

**gold_compression (5089 chars):**

> Cuba, tên gọi chính thức là Cộng hòa Cuba (tiếng Tây Ban Nha: Cuba hay República de Cuba, IPA: [re'puβlika ðe 'kuβa]) là một quốc gia chủ quyền nằm tại khu vực Bắc Mỹ, cụ thể là ở phía bắc Vùng Caribe. Quốc gia này bao gồm đảo Cuba (hòn đảo lớn nhất của quần đảo Đại Antilles, thường được miêu tả có hình dáng giống một con cá sấu đang vươn dài trên biển Caribe), cùng với đảo Thanh Niên (Isla de la Juventud) và nhiều quần đảo, đảo nhỏ và đá ngầm xung quanh. Về mặt vị trí địa lý chiến lược, Cuba nằm ở giao điểm của ba miền biển lớn: Biển Caribe, Vịnh México và Đại Tây Dương. Cuba nằm ở phía nam miền đông Hoa Kỳ (cách bang Florida chỉ khoảng 145 km qua Eo biển Florida) và Bahamas, phía tây Quần đảo Turks và Caicos và Haiti, và phía đông México. Quần đảo Cayman và Jamaica nằm ở phía nam. Khí hậu nhiệt đới gió mùa với hai mùa rõ rệt là mùa khô và mùa mưa, chịu ảnh hưởng của dòng biển Gulf Stream.
> 
> Theo các văn tịch lịch sử thì lịch sử Cuba bắt đầu ngày 28 tháng 10 năm 1492, khi nhà hàng hải Christopher Colombo nhận diện đảo này trong chuyến đi thứ nhất của ông và tuyên bố chủ quyền nhân danh triều đình Tây Ban Nha. (Điều này vẫn còn chưa chắc chắn và là một vấn đề chưa ngã ngũ về mặt học thuật.) Hòn đảo này trước đó đã được thổ dân châu Mỹ, bao gồm bộ tộc Taíno và Ciboney xuất xứ từ Nam Mỹ nhiều thế kỷ trước đến lập nghiệp. Người Taíno và người Ciboney (cũng được viết là Siboney các quốc gia Taino mới) đều là những cư dân biết trồng cấy cùng săn bắn-hái lượm; có thuyết cho rằng kỹ 
> …[cắt]

---

## 14. [extras(unanswerable)] `comp_qa_uit_007015_8x` — doc `viquad:Ả_Rập_Xê_Út`

- split=`train` · domain=`admin` · ctx_chars=23720
- requested=8.0x · target_tok=660 · realized_tok=584 · realized=9.04x · budget_ok=True
- extractive_ratio=0.913 · numbers_preserved=0.050724637681159424 · answerable=False · f1=0.4211
- judge_prediction: `Quân chủ chuyên chế, chế độ độc tài thế tập do hoàng tộc cai trị.`
- human gold answer: `quân chủ chuyên chế`

**query:** Chế độ nhà nước của Ả Rập Xê Út là gì?

**context (600 đầu):**

> Lãnh thổ Ả Rập Xê Út ngày nay khi xưa là 4 khu vực riêng biệt: Hejaz, Najd, bộ phận của Đông Ả Rập (Al-Ahsa) và Nam Ả Rập ('Asir). Vương quốc Ả Rập Xê Út được Ibn Saud thành lập vào năm 1932, "Saud" bắt nguồn từ hoàng tộc Saud, thể hiện quan điểm quốc gia là tài sản cá nhân của hoàng tộc. Ibn Saud thống nhất bốn khu vực thành một quốc gia duy nhất thông qua một loạt cuộc chinh phạt bắt đầu từ năm 1902. Ả Rập Xê Út từ đó là một quốc gia quân chủ chuyên chế, thực chất là một chế độ độc tài thế tập do các dòng Hồi giáo cai trị. Phong trào tôn giáo Wahhabi cực bảo thủ thuộc phái Hồi giáo Sunni đượ

**gold_compression (2762 chars):**

> Ả Rập Xê Út do Ibn Saud thành lập năm 1932, là quốc gia quân chủ chuyên chế, chế độ độc tài thế tập do hoàng tộc cai trị. Phong trào Wahhabi cực bảo thủ thuộc Hồi giáo Sunni là đặc điểm văn hóa nổi bật. Dầu mỏ phát hiện năm 1938 đưa quốc gia này thành nước xuất khẩu dầu lớn nhất, trữ lượng dầu thứ hai, khí đốt thứ sáu thế giới. Dù là nền kinh tế thu nhập cao, quốc gia Ả Rập duy nhất trong G-20, kinh tế Ả Rập Xê Út kém đa dạng, phụ thuộc hoàn toàn dầu khí. Quốc gia chi tiêu quân sự nhiều thứ tư và nhập khẩu vũ khí lớn thứ nhì thế giới.
> 
> Theo Luật Cơ bản 1992, Quran và Sunnah là hiến pháp. Không có chính đảng hay bầu cử quốc gia, bị đánh giá là chế độ chuyên chế, "không tự do". Toàn bộ nam giới thành niên có quyền kiến nghị trực tiếp quốc vương qua majlis (hội nghị bộ lạc). Bản sắc bộ lạc còn mạnh, sheikh bộ lạc duy trì ảnh hưởng lớn. Chính phủ từng bước mở rộng chính trị như lập Hội đồng Cố vấn và Diễn đàn Đối thoại Quốc gia.
> 
> Quyền cai trị hoàng tộc đối diện phản đối từ nhà hoạt động Hồi giáo Sunni, phê bình tự do, cộng đồng Shia tại Vùng Đông và đối thủ bộ lạc. Hoàng tộc phân chia phái hệ dòng dõi, tham vọng, tư tưởng; phái mạnh nhất là 'Sudairi Bảy'. Ranh giới tài sản quốc gia và của cải cá nhân thân vương mập mờ, thường bị cáo buộc tham nhũng.
> 
> Ả Rập Xê Út trao Ulema (thể chế thủ lĩnh tôn giáo, luật gia) vai trò trực tiếp chính phủ. Ulema có ảnh hưởng then chốt quyết định trọng đại, kiểm soát tư pháp, giáo dục, độc quyền đạo đức tôn giáo. Dù quyền lực suy thoái thập niên 1
> …[cắt]

---

## 15. [test] `qa_uit_000182` — doc `viquad:Phần_mềm_giáo_dục`

- split=`test` · domain=`other` · ctx_chars=6746
- task=`short_context_extractive_qa` · is_long_document=False · answer_lang=`vi` · num_passages=8
- answer: `năm 1943` span=[316, 324]
- gold_compression_source=`answer-sentence (human span)` · verification_method=`none`

**query:** Hệ thống the type19 synthetic radar trainer được xây dựng vào thời điểm nào?

**context (600 đầu):**

> Ngay từ đầu những năm 1940, phần cứng và phần mềm đã được sử dụng trong giáo dục và đào tạo, khi đó các nhà nghiên cứu Mỹ phát triển các mô hình tập bay sử dụng máy tính analog để tạo ra các mô phỏng cài đặt trong thiết bị dữ liệu. Một hệ thống thuộc loại này là the type19 synthetic radar trainer được xây dựng vào năm 1943. Từ những cố gắng ban đầu này, trong giai đoạn từ chiến tranh thế giới II đến giữa 1970, phần mềm giáo dục được cài đặt trực tiếp vào phần cứng, thường là các máy tính lớn. Người đi tiên phong trong giai đoạn này là PLATO (1960) được phát triển ở đại học Illinois và TICCIT (

**gold_compression (93 chars):**

> Một hệ thống thuộc loại này là the type19 synthetic radar trainer được xây dựng vào năm 1943.

---

## 16. [test] `qa_uit_017581` — doc `viquad:Lệch_lạc_(xã_hội_học)`

- split=`test` · domain=`other` · ctx_chars=7424
- task=`short_context_extractive_qa` · is_long_document=False · answer_lang=`vi` · num_passages=7
- answer: `trán thấp, cằm, gò má nhô, tai vểnh, nhiều râu tóc và cánh tay dài bất thường trông giống như tổ tiên giống vượn của con người` span=[1536, 1662]
- gold_compression_source=`answer-sentence (human span)` · verification_method=`none`

**query:** Tội phạm được miêu tả trong thuyết tội phạm sinh học như thế nào?

**context (600 đầu):**

> Sự lệch lạc, hay còn gọi là Sự lầm lạc, Hành vi lệch lạc, (tiếng Anh: deviance hoặc deviant behavior) là một khái niệm của xã hội học được định nghĩa là sự vi phạm có nhận thức các tiêu chuẩn hoặc kỳ vọng của một nhóm hay của xã hội. Các tiêu chuẩn văn hóa và kỳ vọng định dạng một dải rộng các hoạt động của con người nên khái niệm sự lệch lạc cũng mang nghĩa rộng tương ứng. Một dạng hiển nhiên của lệch lạc là tội phạm, sự vi phạm các quy phạm được ban hành chính thức thành luật pháp. Ngoài ra nó còn là rất nhiều những dạng không tuân thủ tiêu chuẩn hoặc kỳ vọng khác ở rất nhiều mức độ từ ôn hò

**gold_compression (189 chars):**

> Thuyết này đã mô tả các đặc điểm về thể chất của tội phạm như trán thấp, cằm, gò má nhô, tai vểnh, nhiều râu tóc và cánh tay dài bất thường trông giống như tổ tiên giống vượn của con người.

---

## 17. [test] `qa_uit_002508` — doc `viquad:Dương_cầm`

- split=`test` · domain=`other` · ctx_chars=6070
- task=`short_context_extractive_qa` · is_long_document=False · answer_lang=`vi` · num_passages=7
- answer: `sự yếu ớt trong âm thanh` span=[5183, 5207]
- gold_compression_source=`answer-sentence (human span)` · verification_method=`none`

**query:** Điểm hạn chế của những nhạc cụ phím thời đầu là gì?

**context (600 đầu):**

> Những chiếc dương cầm cổ điển hay còn gọi thông thường là piano cổ điển ngày nay được xây dựng trực tiếp từ những chiếc đàn clavico clavecin (harpsichord) từ khoảng thế kỷ 16 và 17. Khoảng năm 1700, Bartolomeo Cristofori đã thử tạo ra một chiếc đàn harpsichord mà có thể biểu hiện âm nhạc một cách truyền cảm hơn, và đã tạo ra một bộ máy mà các búa gõ vào các dây, khác với đàn harpsichord là dùng quill (dụng cụ gảy đàn bằng ống lông) để gảy. Một đặc trưng lớn khác ở đàn piano thời đầu của ông là cơ cấu búa thoát, nó khiến cho búa tách rời khỏi phím một khi các nốt được đánh lên, và rồi chơi lại 

**gold_compression (133 chars):**

> Một sự tụt hậu trong những nhạc cụ phím đầu tiên, bao gồm cả những chiếc dương cầm vuông đầu tiên, chính là sự yếu ớt trong âm thanh.

---

## 18. [test] `qa_uit_003646` — doc `viquad:Texas`

- split=`test` · domain=`other` · ctx_chars=23809
- task=`long_document_qa` · is_long_document=True · answer_lang=`vi` · num_passages=37
- answer: `11 vùng` span=[4322, 4329]
- gold_compression_source=`answer-sentence (human span)` · verification_method=`none`

**query:** Số lượng vùng sinh thái riêng biệt tại bang Texas là bao nhiêu?

**context (600 đầu):**

> Texas (phát âm: "teksəs" là tiểu bang đông dân thứ hai và có diện tích lớn thứ hai trong số 50 tiểu bang của Hợp chúng quốc Hoa Kỳ, và là tiểu bang lớn nhất trong số 48 tiểu bang liền kề của Hoa Kỳ. Về mặt địa lý, Texas nằm ở vùng Trung Nam của quốc gia, có biên giới quốc tế với các bang Chihuahua, Coahuila, Nuevo León, và Tamaulipas của México; bên trong Hoa Kỳ, Texas có biên giới với tiểu bang New Mexico ở phía tây, Oklahoma ở phía bắc, Arkansas ở phía đông bắc, và Louisiana ở phía đông. Texas có diện tích 696.200 kilômét vuông (268.800 sq mi) với 26,1 triệu cư dân.
> 
> Houston là thành phố lớn

**gold_compression (210 chars):**

> Texas có 10 vùng khí hậu, 14 vùng đất đai và 11 vùng sinh thái riêng biệt, do vậy việc phân loại vùng miền trở nên mơ hồ với những khác biệt về đất, địa hình, địa chất, lượng mưa, quần thể thực vật và động vật.

---

## 19. [qa] `qa_uit_005522` — doc `viquad:Alexandros_Đại_đế`

- split=`train` · domain=`other` · ctx_chars=23682
- task=`long_document_qa` · is_long_document=True · answer_lang=`vi` · num_passages=24
- answer: `chinh phạt gần như toàn bộ thế giới mà con người thời đó biết đến trước khi qua đời` span=[523, 606]

**query:** Vì sao ông được xem là một trong những vị tướng thành công nhất trong lịch sử?

**context (600 đầu):**

> Alexandros III của Macedonia, được biết rộng rãi với cái tên Alexandros Đại đế, (tiếng Hy Lạp: Μέγας Αλέξανδρος Megas Alexandros, tiếng Latinh: Alexander Magnus) (tháng 7 năm 356 TCN – 11 tháng 6 năm 323 TCN), là Quốc vương thứ 14 của nhà Argead ở Vương quốc Macedonia (336 – 323 TCN), nhưng ít dành thời gian cho việc trị quốc tại quê nhà Macedonia. Trong suốt Triều đại của mình, ông chủ yếu dành thời gian cho quân sự, các cuộc chinh phạt, và được xem là một trong những vị tướng thành công nhất trong lịch sử, người đã chinh phạt gần như toàn bộ thế giới mà con người thời đó biết đến trước khi q

---

## 20. [qa] `qa_uit_005924` — doc `viquad:Đại_học_Chicago`

- split=`train` · domain=`other` · ctx_chars=7174
- task=`short_context_extractive_qa` · is_long_document=False · answer_lang=`vi` · num_passages=8
- answer: `thay thế cho viện đại học có cùng tên gọi của những người theo phái Baptist, vốn đã đóng cửa vào năm 1886` span=[1563, 1668]

**query:** Vì sao Viện Đại học Chicago ra đời?

**context (600 đầu):**

> Viện Đại học Chicago bao gồm Trường Đại học (the College), nhiều chương trình sau đại học và ủy ban liên ngành khác nhau được tổ chức thành bốn phân khoa, sáu trường chuyên nghiệp, và một trường giáo dục thường xuyên. Viện đại học có tổng cộng khoảng 15.000 sinh viên, trong đó chừng 5.000 sinh viên theo học ở Trường Đại học. Viện Đại học Chicago nhiều năm liền được xếp vào một trong 10 viện đại học hàng đầu thế giới; và được xếp thứ năm cùng với Viện Đại học Stanford trong "Bảng xếp hạng những viện đại học tốt nhất nước" năm 2014 của U.S. News & World Report.
> 
> Các học giả của Viện Đại học Chic
