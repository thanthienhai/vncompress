#!/usr/bin/env python3
"""
Build VCC-Bench dataset from raw Vietnamese texts.
Creates 5 tasks with real data and quality validation.
"""
import json
import os
import sys
import time
import random
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'benchmark')
os.makedirs(DATA_DIR, exist_ok=True)

random.seed(42)

# ============================================================================
# VIETNAMESE LEGAL TEXTS (public domain)
# ============================================================================

VIETNAMESE_LEGAL_TEXTS = {
    "hien_phap_2013": {
        "title": "Hiến pháp nước Cộng hòa Xã hội Chủ nghĩa Việt Nam 2013",
        "chapters": {
            "Chương I: Chế độ chính trị": [
                "Điều 1: Nước Cộng hòa xã hội chủ nghĩa Việt Nam là một nước độc lập, có chủ quyền, thống nhất và toàn vẹn lãnh thổ, bao gồm đất liền, hải đảo, vùng biển và vùng trời.",
                "Điều 2: Nhà nước Cộng hòa xã hội chủ nghĩa Việt Nam là nhà nước pháp quyền xã hội chủ nghĩa của Nhân dân, do Nhân dân, vì Nhân dân. Nước Cộng hòa xã hội chủ nghĩa Việt Nam do Nhân dân làm chủ; tất cả quyền lực nhà nước thuộc về Nhân dân mà nền tảng là liên minh giữa giai cấp công nhân với giai cấp nông dân và đội ngũ trí thức.",
                "Điều 3: Nhà nước bảo đảm và phát huy quyền làm chủ của Nhân dân; công nhận, tôn trọng, bảo vệ và bảo đảm quyền con người, quyền công dân; thực hiện mục tiêu dân giàu, nước mạnh, dân chủ, công bằng, văn minh, mọi người có cuộc sống ấm no, tự do, hạnh phúc, có điều kiện phát triển toàn diện.",
                "Điều 4: Đảng Cộng sản Việt Nam - Đội tiên phong của giai cấp công nhân, đồng thời là đội tiên phong của Nhân dân lao động và của dân tộc Việt Nam, đại biểu trung thành lợi ích của giai cấp công nhân, Nhân dân lao động và của cả dân tộc, lấy chủ nghĩa Mác - Lê nin và tư tưởng Hồ Chí Minh làm nền tảng tư tưởng, là lực lượng lãnh đạo Nhà nước và xã hội.",
                "Điều 5: Nước Cộng hòa xã hội chủ nghĩa Việt Nam là quốc gia thống nhất của các dân tộc cùng sinh sống trên đất nước Việt Nam. Các dân tộc bình đẳng, đoàn kết, tôn trọng và giúp nhau cùng phát triển; nghiêm cấm mọi hành vi kỳ thị, chia rẽ dân tộc."
            ],
            "Chương II: Quyền con người, quyền và nghĩa vụ cơ bản của công dân": [
                "Điều 14: Ở nước Cộng hòa xã hội chủ nghĩa Việt Nam, các quyền con người, quyền công dân về chính trị, dân sự, kinh tế, văn hóa, xã hội được công nhận, tôn trọng, bảo vệ, bảo đảm theo Hiến pháp và pháp luật. Quyền con người, quyền công dân chỉ có thể bị hạn chế theo quy định của luật trong trường hợp cần thiết vì lý do quốc phòng, an ninh quốc gia, trật tự, an toàn xã hội, đạo đức xã hội, sức khỏe của cộng đồng.",
                "Điều 15: Quyền công dân không tách rời nghĩa vụ công dân. Mọi người có nghĩa vụ tôn trọng quyền của người khác. Công dân có trách nhiệm thực hiện nghĩa vụ đối với Nhà nước và xã hội. Việc thực hiện quyền con người, quyền công dân không được xâm phạm lợi ích quốc gia, dân tộc, quyền và lợi ích hợp pháp của người khác.",
                "Điều 16: Mọi người đều bình đẳng trước pháp luật. Không ai bị phân biệt đối xử trong đời sống chính trị, dân sự, kinh tế, văn hóa, xã hội.",
                "Điều 19: Mọi người có quyền sống. Tính mạng con người được pháp luật bảo hộ. Không ai bị tước đoạt tính mạng trái luật.",
                "Điều 20: Mọi người có quyền bất khả xâm phạm về thân thể, được pháp luật bảo hộ về sức khỏe, danh dự và nhân phẩm; không bị tra tấn, bạo lực, truy bức, nhục hình hay bất kỳ hình thức đối xử nào khác xâm phạm thân thể, sức khỏe, xúc phạm danh dự, nhân phẩm."
            ],
            "Chương III: Kinh tế, Xã hội, Văn hóa, Giáo dục, Khoa học, Công nghệ và Môi trường": [
                "Điều 50: Nước Cộng hòa xã hội chủ nghĩa Việt Nam xây dựng nền kinh tế độc lập, tự chủ, phát huy nội lực, hội nhập, hợp tác quốc tế, gắn kết chặt chẽ với phát triển văn hóa, thực hiện tiến bộ và công bằng xã hội, bảo vệ môi trường, thực hiện công nghiệp hóa, hiện đại hóa đất nước.",
                "Điều 51: Nền kinh tế Việt Nam là nền kinh tế thị trường định hướng xã hội chủ nghĩa với nhiều hình thức sở hữu, nhiều thành phần kinh tế; kinh tế nhà nước giữ vai trò chủ đạo.",
                "Điều 52: Nhà nước xây dựng và hoàn thiện thể chế kinh tế, điều tiết nền kinh tế trên cơ sở tôn trọng các quy luật thị trường; thực hiện phân công, phân cấp, phân quyền trong quản lý nhà nước; thúc đẩy liên kết kinh tế vùng, bảo đảm tính thống nhất của nền kinh tế quốc dân.",
                "Điều 58: Nhà nước, xã hội đầu tư phát triển sự nghiệp bảo vệ, chăm sóc sức khỏe của Nhân dân, thực hiện bảo hiểm y tế toàn dân, có chính sách ưu tiên chăm sóc sức khỏe cho đồng bào dân tộc thiểu số, đồng bào ở miền núi, hải đảo và vùng có điều kiện kinh tế - xã hội đặc biệt khó khăn.",
                "Điều 61: Nhà nước ưu tiên đầu tư và thu hút các nguồn đầu tư khác cho giáo dục; chăm lo giáo dục mầm non; bảo đảm giáo dục tiểu học là bắt buộc, Nhà nước không thu học phí; từng bước phổ cập giáo dục trung học; phát triển giáo dục đại học, giáo dục nghề nghiệp; thực hiện chính sách học bổng, học phí hợp lý."
            ],
            "Chương IV: Bảo vệ Tổ quốc": [
                "Điều 64: Bảo vệ Tổ quốc Việt Nam xã hội chủ nghĩa là sự nghiệp của toàn dân. Nhà nước củng cố và tăng cường nền quốc phòng toàn dân và an ninh nhân dân mà nòng cốt là lực lượng vũ trang nhân dân; phát huy sức mạnh tổng hợp của đất nước để bảo vệ vững chắc Tổ quốc, góp phần bảo vệ hòa bình ở khu vực và trên thế giới.",
                "Điều 65: Lực lượng vũ trang nhân dân tuyệt đối trung thành với Tổ quốc, Nhân dân, với Đảng và Nhà nước, có nhiệm vụ bảo vệ độc lập, chủ quyền, thống nhất, toàn vẹn lãnh thổ của Tổ quốc, an ninh quốc gia và trật tự, an toàn xã hội; bảo vệ Nhân dân, Đảng, Nhà nước và chế độ xã hội chủ nghĩa."
            ],
            "Chương V: Quốc hội": [
                "Điều 69: Quốc hội là cơ quan đại biểu cao nhất của Nhân dân, cơ quan quyền lực nhà nước cao nhất của nước Cộng hòa xã hội chủ nghĩa Việt Nam. Quốc hội thực hiện quyền lập hiến, quyền lập pháp, quyết định các vấn đề quan trọng của đất nước và giám sát tối cao đối với hoạt động của Nhà nước.",
                "Điều 70: Quốc hội có những nhiệm vụ và quyền hạn: Làm Hiến pháp và sửa đổi Hiến pháp; làm luật và sửa đổi luật; Quyết định mục tiêu, chỉ tiêu, chính sách, nhiệm vụ cơ bản phát triển kinh tế - xã hội của đất nước; Quyết định chính sách cơ bản về tài chính, tiền tệ quốc gia."
            ]
        }
    },
    "luat_bao_ve_moi_truong_2020": {
        "title": "Luật Bảo vệ Môi trường 2020",
        "chapters": {
            "Chương I: Những quy định chung": [
                "Điều 1: Luật này quy định về hoạt động bảo vệ môi trường; quyền, nghĩa vụ và trách nhiệm của cơ quan, tổ chức, cộng đồng dân cư, hộ gia đình và cá nhân trong hoạt động bảo vệ môi trường.",
                "Điều 4: Bảo vệ môi trường là quyền, nghĩa vụ và trách nhiệm của mọi cơ quan, tổ chức, cộng đồng dân cư, hộ gia đình và cá nhân. Hoạt động bảo vệ môi trường phải được tiến hành thường xuyên, công khai, minh bạch; ưu tiên dự báo, phòng ngừa ô nhiễm, sự cố, suy thoái môi trường. Bảo vệ môi trường gắn kết hài hòa với phát triển kinh tế, an sinh xã hội, bảo đảm quyền trẻ em, thúc đẩy bình đẳng giới và phát triển bền vững."
            ],
            "Chương II: Bảo vệ môi trường đối với các hoạt động sản xuất, kinh doanh, dịch vụ": [
                "Điều 25: Dự án đầu tư, cơ sở sản xuất, kinh doanh, dịch vụ có phát sinh chất thải, tác động xấu đến môi trường phải có hệ thống quan trắc, giám sát môi trường theo quy định của pháp luật. Chất thải phải được quản lý trong toàn bộ quá trình phát sinh, thu gom, lưu giữ, vận chuyển, xử lý và tiêu hủy.",
                "Điều 34: Khu công nghiệp, khu chế xuất, khu công nghệ cao, cụm công nghiệp phải có hệ thống hạ tầng bảo vệ môi trường đồng bộ; có hệ thống thu gom, xử lý nước thải tập trung; có thiết bị quan trắc nước thải tự động liên tục; có phương án phòng ngừa, ứng phó sự cố môi trường."
            ],
            "Chương III: Ứng phó với biến đổi khí hậu": [
                "Điều 55: Nội dung thích ứng với biến đổi khí hậu bao gồm đánh giá tác động, tính dễ bị tổn thương, rủi ro, tổn thất và thiệt hại do biến đổi khí hậu đối với các lĩnh vực, khu vực và cộng đồng dân cư. Trên cơ sở đó, triển khai các giải pháp thích ứng, giảm nhẹ rủi ro, tổn thất và thiệt hại do biến đổi khí hậu.",
                "Điều 56: Nội dung giảm nhẹ phát thải khí nhà kính bao gồm xây dựng và thực hiện kế hoạch, biện pháp giảm nhẹ phát thải khí nhà kính phù hợp với điều kiện kinh tế - xã hội; phát triển và ứng dụng công nghệ phát thải các-bon thấp, thân thiện với môi trường.",
                "Điều 57: Bảo vệ tầng ô-dôn bao gồm quản lý và loại trừ các chất làm suy giảm tầng ô-dôn, các chất gây hiệu ứng nhà kính được kiểm soát trong khuôn khổ điều ước quốc tế về bảo vệ tầng ô-dôn mà Việt Nam là thành viên."
            ]
        }
    },
    "luat_doanh_nghiep_2020": {
        "title": "Luật Doanh nghiệp 2020",
        "chapters": {
            "Chương I: Những quy định chung": [
                "Điều 1: Luật này quy định về việc thành lập, tổ chức quản lý, tổ chức lại, giải thể và hoạt động có liên quan của doanh nghiệp, bao gồm công ty trách nhiệm hữu hạn, công ty cổ phần, công ty hợp danh và doanh nghiệp tư nhân; quy định về nhóm công ty.",
                "Điều 4: Doanh nghiệp là tổ chức có tên riêng, có tài sản, có trụ sở giao dịch, được thành lập hoặc đăng ký thành lập theo quy định của pháp luật nhằm mục đích kinh doanh. Nhà nước công nhận sự tồn tại lâu dài và phát triển của các loại hình doanh nghiệp quy định tại Luật này.",
                "Điều 7: Doanh nghiệp có quyền tự do kinh doanh ngành, nghề mà luật không cấm. Doanh nghiệp có quyền tự chủ kinh doanh và tự chịu trách nhiệm về kết quả kinh doanh."
            ],
            "Chương V: Công ty cổ phần": [
                "Điều 111: Công ty cổ phần là doanh nghiệp, trong đó vốn điều lệ được chia thành nhiều phần bằng nhau gọi là cổ phần. Cổ đông có thể là tổ chức, cá nhân; số lượng cổ đông tối thiểu là 03 và không hạn chế số lượng tối đa. Cổ đông chỉ chịu trách nhiệm về các khoản nợ và nghĩa vụ tài sản khác của doanh nghiệp trong phạm vi số vốn đã góp vào doanh nghiệp.",
                "Điều 112: Công ty cổ phần có quyền phát hành cổ phần, trái phiếu và các loại chứng khoán khác của công ty. Các loại cổ phần bao gồm cổ phần phổ thông và cổ phần ưu đãi. Người sở hữu cổ phần phổ thông là cổ đông phổ thông. Cổ phần ưu đãi gồm các loại: cổ phần ưu đãi biểu quyết, cổ phần ưu đãi cổ tức, cổ phần ưu đãi hoàn lại và cổ phần ưu đãi khác."
            ]
        }
    }
}

# ============================================================================
# MULTI-TURN CONVERSATIONS (Vietnamese, realistic)
# ============================================================================

VIETNAMESE_CONVERSATIONS = [
    {
        "scenario": "customer_service",
        "turns": [
            {"speaker": "Khách hàng", "text": "Chào bạn, tôi muốn đổi trả sản phẩm điện thoại tôi mua tuần trước vì màn hình bị lỗi cảm ứng."},
            {"speaker": "Nhân viên", "text": "Dạ chào anh/chị. Anh/chị cho em xin mã đơn hàng và tên người mua để em kiểm tra trong hệ thống ạ."},
            {"speaker": "Khách hàng", "text": "Mã đơn hàng là DH20260701001, tên tôi là Nguyễn Văn An. Tôi mua iPhone 15 Pro Max ngày 01/07/2026."},
            {"speaker": "Nhân viên", "text": "Dạ em đã kiểm tra. Đơn hàng của anh/chị đang trong thời hạn đổi trả 14 ngày. Anh/chị có thể mang sản phẩm đến cửa hàng gần nhất để được kiểm tra và đổi mới ạ."},
            {"speaker": "Khách hàng", "text": "Nhưng cửa hàng gần nhất cách nhà tôi 50km. Có cách nào khác không?"},
            {"speaker": "Nhân viên", "text": "Dạ vậy anh/chị có thể gửi sản phẩm qua đường bưu điện. Bên em sẽ hỗ trợ miễn phí vận chuyển hai chiều cho đơn đổi trả trong 14 ngày ạ. Thời gian xử lý khoảng 3-5 ngày làm việc."},
            {"speaker": "Khách hàng", "text": "Vậy được. Bạn hướng dẫn tôi cách gửi nhé. Tôi cần chuẩn bị những gì?"},
            {"speaker": "Nhân viên", "text": "Anh/chị cần đóng gói sản phẩm đầy đủ phụ kiện, hộp, sạc, cáp như lúc mua. Sau đó liên hệ tổng đài 1900xxxx để đặt lịch lấy hàng tận nơi. Nhân viên bưu điện sẽ đến lấy trong 24h ạ."},
            {"speaker": "Khách hàng", "text": "Cảm ơn bạn. Tôi sẽ gọi tổng đài ngay. Một câu nữa: khi nào tôi nhận được sản phẩm mới?"},
            {"speaker": "Nhân viên", "text": "Dạ sau khi bên em nhận được sản phẩm, bộ phận kỹ thuật sẽ kiểm tra trong 24h. Nếu đúng lỗi từ nhà sản xuất, em sẽ gửi sản phẩm mới cho anh/chị trong vòng 48h tiếp theo. Tổng thời gian khoảng 4-5 ngày ạ."},
        ]
    },
    {
        "scenario": "recruitment_interview",
        "turns": [
            {"speaker": "Nhà tuyển dụng", "text": "Chào bạn, cảm ơn bạn đã đến phỏng vấn hôm nay. Bạn có thể giới thiệu ngắn gọn về bản thân và kinh nghiệm làm việc không?"},
            {"speaker": "Ứng viên", "text": "Chào anh/chị. Tôi tên là Trần Thị Bình, tốt nghiệp Đại học Bách Khoa Hà Nội chuyên ngành Khoa học Máy tính năm 2022. Tôi có 4 năm kinh nghiệm làm việc với Python và các framework deep learning như PyTorch và TensorFlow."},
            {"speaker": "Nhà tuyển dụng", "text": "Trong CV bạn có đề cập đến dự án xây dựng hệ thống chatbot cho ngân hàng. Bạn có thể mô tả chi tiết hơn về dự án này không?"},
            {"speaker": "Ứng viên", "text": "Đó là dự án tôi làm tại công ty ABC Tech. Chúng tôi xây dựng chatbot hỗ trợ khách hàng cho một ngân hàng thương mại, xử lý khoảng 10,000 yêu cầu mỗi ngày. Tôi phụ trách phần NLP pipeline, bao gồm intent classification, entity extraction, và dialogue management sử dụng Rasa framework."},
            {"speaker": "Nhà tuyển dụng", "text": "Thách thức lớn nhất bạn gặp phải trong dự án đó là gì?"},
            {"speaker": "Ứng viên", "text": "Thách thức lớn nhất là xử lý ngôn ngữ tiếng Việt với các biến thể địa phương và teencode. Chúng tôi phải xây dựng bộ normalization pipeline riêng để chuẩn hóa input trước khi đưa vào model, và fine-tune PhoBERT để cải thiện accuracy intent classification từ 78% lên 93%."},
            {"speaker": "Nhà tuyển dụng", "text": "Rất ấn tượng. Bạn có câu hỏi gì cho chúng tôi không?"},
            {"speaker": "Ứng viên", "text": "Tôi muốn hỏi về lộ trình phát triển của team trong 12 tháng tới, và công ty có hỗ trợ nhân viên tham gia các hội thảo, khóa học nâng cao không ạ?"},
        ]
    },
    {
        "scenario": "healthcare_consultation",
        "turns": [
            {"speaker": "Bệnh nhân", "text": "Chào bác sĩ, tôi bị đau đầu kéo dài khoảng 1 tuần nay, kèm theo chóng mặt và mất ngủ. Tôi rất lo lắng không biết có vấn đề gì nghiêm trọng không."},
            {"speaker": "Bác sĩ", "text": "Chào bạn. Bạn có thể mô tả thêm về cơn đau không? Đau ở vị trí nào trên đầu, cường độ ra sao, và có yếu tố nào làm cơn đau tăng lên không?"},
            {"speaker": "Bệnh nhân", "text": "Đau chủ yếu ở vùng thái dương hai bên, cảm giác như bị bóp chặt. Cơn đau thường nặng hơn vào buổi chiều sau khi làm việc với máy tính. Tôi làm lập trình viên nên ngồi trước màn hình 8-10 tiếng mỗi ngày."},
            {"speaker": "Bác sĩ", "text": "Nghe mô tả của bạn, có thể đây là đau đầu do căng thẳng (tension headache) kết hợp với mỏi mắt do sử dụng màn hình kéo dài. Bạn có tiền sử bệnh gì không? Huyết áp của bạn thế nào?"},
            {"speaker": "Bệnh nhân", "text": "Tôi không có bệnh nền. Cách đây 1 tháng tôi có đo huyết áp thì ở mức 125/85, bác sĩ nói hơi cao nhẹ."},
            {"speaker": "Bác sĩ", "text": "Tôi khuyên bạn nên: (1) Nghỉ giải lao mỗi 45-60 phút khi làm việc, áp dụng quy tắc 20-20-20 (mỗi 20 phút nhìn xa 20 feet trong 20 giây); (2) Tập thể dục nhẹ nhàng 30 phút mỗi ngày; (3) Uống đủ 2 lít nước mỗi ngày. Tôi sẽ kê đơn thuốc giảm đau Paracetamol 500mg, uống khi đau nhiều. Nếu sau 1 tuần không cải thiện, bạn nên đến khám trực tiếp để làm thêm xét nghiệm."},
        ]
    },
    {
        "scenario": "travel_planning",
        "turns": [
            {"speaker": "Du khách", "text": "Chào bạn, tôi đang lên kế hoạch du lịch Đà Nẵng - Hội An 4 ngày 3 đêm vào giữa tháng 8. Bạn có thể tư vấn lịch trình giúp tôi không?"},
            {"speaker": "Tư vấn viên", "text": "Dạ chào anh/chị. Đà Nẵng - Hội An tháng 8 đang là mùa cao điểm du lịch, thời tiết nắng đẹp, ít mưa. Em đề xuất lịch trình: Ngày 1 khám phá Đà Nẵng (Bà Nà Hills, Cầu Vàng, biển Mỹ Khê). Ngày 2 tham quan bán đảo Sơn Trà, Ngũ Hành Sơn. Ngày 3-4 ở Hội An (phố cổ, làng gốm Thanh Hà, biển Cửa Đại). Anh/chị thấy thế nào ạ?"},
            {"speaker": "Du khách", "text": "Lịch trình nghe hấp dẫn đấy. Nhưng tôi đi cùng gia đình có trẻ nhỏ 5 tuổi, nên cần lịch trình nhẹ nhàng hơn. Với lại tôi muốn biết chi phí ước tính."},
            {"speaker": "Tư vấn viên", "text": "Dạ em hiểu rồi. Với gia đình có trẻ nhỏ em đề xuất giảm bớt Bà Nà Hills vì cáp treo và đông đúc có thể không phù hợp. Thay vào đó ngày 1 tham quan biển Mỹ Khê buổi sáng, chiều đi Công viên Châu Á. Chi phí ước tính cho 4 người: vé máy bay khứ hồi khoảng 8-10 triệu, khách sạn 3 sao 3 đêm khoảng 4-5 triệu, ăn uống và vé tham quan khoảng 6-8 triệu. Tổng khoảng 18-23 triệu ạ."},
        ]
    },
    {
        "scenario": "academic_consultation",
        "turns": [
            {"speaker": "Sinh viên", "text": "Thưa thầy, em đang làm khóa luận tốt nghiệp về ứng dụng Transformer trong xử lý ngôn ngữ tiếng Việt. Em phân vân giữa hai hướng: phân loại văn bản pháp luật và tóm tắt tin tức tự động. Thầy có thể tư vấn giúp em không ạ?"},
            {"speaker": "Giảng viên", "text": "Cả hai hướng đều có giá trị nghiên cứu. Tuy nhiên, phân loại văn bản pháp luật sẽ có tính ứng dụng cao hơn vì hiện nay các cơ quan nhà nước đang số hóa mạnh mẽ. Em có thể kết hợp với Legal-BERT hoặc các pre-trained model cho legal domain. Còn tóm tắt tin tức thì có nhiều nghiên cứu rồi, novelty sẽ thấp hơn."},
            {"speaker": "Sinh viên", "text": "Dạ vậy em chọn hướng phân loại văn bản pháp luật. Em dự định dùng PhoBERT làm base model và fine-tune trên bộ dữ liệu văn bản pháp luật tiếng Việt. Thầy có gợi ý gì về phương pháp cải thiện accuracy không ạ?"},
            {"speaker": "Giảng viên", "text": "Em nên thử 3 hướng: (1) Data augmentation bằng cách sinh thêm dữ liệu từ các điều luật có cấu trúc tương tự; (2) Hierarchical classification - phân loại theo chương trước rồi mới đến điều khoản cụ thể; (3) Contrastive learning để model phân biệt tốt hơn giữa các nhóm luật gần nhau. Nhớ làm ablation study để đánh giá đóng góp của từng thành phần."},
        ]
    },
    {
        "scenario": "banking_support",
        "turns": [
            {"speaker": "Khách hàng", "text": "Xin chào, tôi muốn hỏi về thủ tục vay vốn mua nhà tại ngân hàng. Tôi cần vay khoảng 2 tỷ đồng trong thời hạn 20 năm."},
            {"speaker": "Tư vấn viên", "text": "Dạ chào anh/chị. Ngân hàng chúng tôi đang có gói vay mua nhà với lãi suất ưu đãi 7.5%/năm trong 2 năm đầu, sau đó thả nổi theo lãi suất thị trường. Anh/chị cần chuẩn bị: CMND/CCCD, sổ hộ khẩu, hợp đồng lao động, sao kê lương 6 tháng gần nhất, và giấy tờ liên quan đến căn nhà định mua."},
            {"speaker": "Khách hàng", "text": "Thu nhập của tôi khoảng 35 triệu/tháng, vợ tôi 20 triệu/tháng. Chúng tôi có thể vay tối đa bao nhiêu và thời gian giải ngân thế nào?"},
            {"speaker": "Tư vấn viên", "text": "Dạ với tổng thu nhập 55 triệu/tháng, anh/chị có thể vay tối đa khoảng 3.2 tỷ đồng. Khoản vay 2 tỷ hoàn toàn khả thi. Thời gian xét duyệt hồ sơ khoảng 5-7 ngày làm việc, sau đó giải ngân trong 2-3 ngày tiếp theo. Anh/chị có muốn đặt lịch hẹn gặp chuyên viên tín dụng để được tư vấn chi tiết hơn không ạ?"},
            {"speaker": "Khách hàng", "text": "Vâng, tôi muốn đặt lịch vào chiều thứ Sáu tuần này."},
        ]
    },
    {
        "scenario": "online_shopping_dispute",
        "turns": [
            {"speaker": "Người mua", "text": "Tôi đặt mua chiếc laptop Dell XPS 15 trên sàn thương mại điện tử ngày 05/07, nhưng khi nhận hàng hôm qua thì sản phẩm bị trầy xước và seal hộp đã bị rách. Tôi yêu cầu đổi trả nhưng người bán không đồng ý."},
            {"speaker": "Hỗ trợ sàn TMĐT", "text": "Dạ chào anh/chị. Em rất tiếc về trải nghiệm không tốt này. Anh/chị cho em xin mã đơn hàng và hình ảnh sản phẩm khi nhận hàng để em kiểm tra và hỗ trợ ạ."},
            {"speaker": "Người mua", "text": "Mã đơn hàng SP20260705-8892. Tôi đã chụp ảnh lúc mở hộp, có video unboxing đầy đủ. Tôi gửi link ảnh qua chat này được không?"},
            {"speaker": "Hỗ trợ sàn TMĐT", "text": "Dạ vâng, anh/chị gửi link ảnh qua đây ạ. Nếu có video unboxing thì càng tốt vì đó là bằng chứng quan trọng. Em sẽ mở khiếu nại chính thức và tạm giữ tiền thanh toán cho đến khi giải quyết xong."},
            {"speaker": "Người mua", "text": "Đây là link: https://cloud.example.com/unboxing-dell-xps15. Khi nào tôi nhận được tiền hoàn lại?"},
            {"speaker": "Hỗ trợ sàn TMĐT", "text": "Dạ em đã nhận được bằng chứng. Em sẽ gửi yêu cầu lên bộ phận giải quyết tranh chấp. Thời gian xử lý thông thường là 3-5 ngày làm việc. Nếu bằng chứng hợp lệ, tiền sẽ được hoàn về tài khoản của anh/chị trong vòng 24h sau khi có kết luận. Em sẽ cập nhật trạng thái qua tin nhắn SMS và email ạ."},
        ]
    },
    {
        "scenario": "real_estate_consultation",
        "turns": [
            {"speaker": "Người mua", "text": "Chào anh, tôi đang tìm mua một căn hộ chung cư tại quận 2, TP.HCM, diện tích khoảng 70-80m2, giá dưới 3 tỷ. Anh có căn nào phù hợp không?"},
            {"speaker": "Môi giới", "text": "Chào chị. Hiện bên em đang có một căn hộ tại dự án The Sun Avenue quận 2, diện tích 75m2, 2 phòng ngủ, 2 WC, view sông Sài Gòn, nội thất cao cấp. Giá chủ nhà đang chào 2.85 tỷ, có thương lượng. Căn này đã có sổ hồng, pháp lý rõ ràng."},
            {"speaker": "Người mua", "text": "Nghe hấp dẫn đấy. Nhưng tôi lo về tiến độ bàn giao và phí quản lý hàng tháng. Anh cho tôi biết thêm chi tiết được không?"},
            {"speaker": "Môi giới", "text": "Căn hộ đã bàn giao từ tháng 3/2026, khách vào ở ngay được. Phí quản lý 15,000đ/m2/tháng, tức khoảng 1.125 triệu/tháng. Phí gửi xe ô tô 1.2 triệu/tháng, xe máy 150,000đ/tháng. Tiện ích nội khu: hồ bơi, gym, công viên, siêu thị mini, trường mầm non. Chị có muốn đi xem nhà thực tế vào cuối tuần này không ạ?"},
        ]
    },
    {
        "scenario": "tech_support_remote",
        "turns": [
            {"speaker": "Khách hàng", "text": "Máy tính của tôi tự nhiên bị treo liên tục, màn hình xanh xuất hiện và khởi động lại. Tôi đã thử khởi động lại nhiều lần nhưng không cải thiện."},
            {"speaker": "Kỹ thuật viên", "text": "Chào anh/chị. Màn hình xanh thường liên quan đến lỗi driver hoặc phần cứng. Anh/chị cho em biết mã lỗi hiển thị trên màn hình xanh được không ạ? Mã thường có dạng 0x00000..."},
            {"speaker": "Khách hàng", "text": "Mã lỗi là MEMORY_MANAGEMENT 0x0000001A. Máy tôi dùng Windows 11, RAM 16GB, ổ SSD 512GB."},
            {"speaker": "Kỹ thuật viên", "text": "Mã lỗi MEMORY_MANAGEMENT thường do lỗi RAM hoặc driver. Anh/chị thử các bước: (1) Chạy Windows Memory Diagnostic: bấm Windows+R, gõ 'mdsched.exe', chọn Restart now; (2) Cập nhật driver chipset và card đồ họa từ trang chủ nhà sản xuất; (3) Kiểm tra xem có phần mềm diệt virus nào đang conflict không. Nếu vẫn lỗi, có thể RAM bị lỗi vật lý, cần thay thế."},
            {"speaker": "Khách hàng", "text": "OK tôi sẽ thử. Nếu RAM bị lỗi thật thì thay thế có phức tạp không?"},
            {"speaker": "Kỹ thuật viên", "text": "Không quá phức tạp ạ. Nếu là laptop, một số dòng RAM hàn chết trên mainboard thì cần thợ chuyên nghiệp. Nếu là PC hoặc laptop có khe cắm rời, anh/chị có thể tự thay được. Bên em có dịch vụ kiểm tra và thay thế tại nhà với chi phí 300,000đ/lần kiểm tra, giá RAM tùy loại từ 800,000đ đến 2,500,000đ."},
        ]
    },
    {
        "scenario": "restaurant_booking",
        "turns": [
            {"speaker": "Khách hàng", "text": "Chào nhà hàng, tôi muốn đặt bàn cho 8 người vào tối thứ Bảy tuần này, lúc 19:00."},
            {"speaker": "Nhà hàng", "text": "Dạ chào anh/chị. Cuối tuần nhà hàng khá đông, để em kiểm tra availability. Anh/chị muốn ngồi trong phòng máy lạnh hay ngoài sân vườn ạ? Ngoài ra anh/chị có yêu cầu gì đặc biệt về món ăn không?"},
            {"speaker": "Khách hàng", "text": "Cho tôi phòng riêng máy lạnh nhé. Trong nhóm có 1 người ăn chay và 1 người dị ứng hải sản. Nhà hàng có set menu cho nhóm 8 người không?"},
            {"speaker": "Nhà hàng", "text": "Dạ có ạ. Bên em có set menu Tiệc Gia Đình 8 người giá 1,200,000đ/người, bao gồm: khai vị 3 món, món chính 5 món (có thể điều chỉnh riêng phần chay và không hải sản), tráng miệng và đồ uống. Em đã giữ phòng VIP 2 cho anh/chị. Anh/chị xác nhận đặt bàn giúp em và để lại số điện thoại để em gửi tin nhắn xác nhận ạ."},
        ]
    },
    {
        "scenario": "flight_booking_change",
        "turns": [
            {"speaker": "Hành khách", "text": "Tôi cần đổi vé máy bay từ Hà Nội đi Phú Quốc ngày 20/07 sang ngày 22/07. Mã đặt chỗ là VN1234."},
            {"speaker": "Tổng đài", "text": "Dạ chào quý khách. Em kiểm tra thông tin đặt chỗ. Vé của quý khách là vé hạng phổ thông tiêu chuẩn, có thể đổi ngày nhưng sẽ có phí đổi vé 500,000đ cộng chênh lệch giá vé nếu có. Ngày 22/07 còn 12 chỗ trống, giá vé cao hơn 300,000đ so với vé cũ. Tổng chi phí đổi vé là 800,000đ. Quý khách có đồng ý không ạ?"},
            {"speaker": "Hành khách", "text": "Chấp nhận được. Bạn đổi giúp tôi nhé. Tôi thanh toán phí đổi vé như thế nào?"},
            {"speaker": "Tổng đài", "text": "Dạ em sẽ gửi link thanh toán qua email đăng ký của quý khách. Sau khi thanh toán, vé mới sẽ được gửi trong vòng 15 phút. Quý khách có muốn chọn chỗ ngồi trước không ạ? Vé mới cho phép chọn chỗ miễn phí trước 24h bay."},
            {"speaker": "Hành khách", "text": "Cho tôi chọn ghế cửa sổ, hàng 15 nếu có."},
        ]
    },
    {
        "scenario": "property_insurance_claim",
        "turns": [
            {"speaker": "Người yêu cầu", "text": "Xin chào, nhà tôi bị ngập nước do mưa lớn tuần trước, hư hỏng nhiều đồ đạc. Tôi có mua bảo hiểm nhà ở gói toàn diện của công ty. Tôi muốn làm thủ tục yêu cầu bồi thường."},
            {"speaker": "Bảo hiểm", "text": "Dạ chào anh/chị, em rất tiếc về sự cố này. Anh/chị cho em xin số hợp đồng bảo hiểm và mô tả sơ bộ thiệt hại để em mở hồ sơ bồi thường ạ."},
            {"speaker": "Người yêu cầu", "text": "Số HĐ: BHN-2026-08921. Thiệt hại gồm: sàn gỗ phòng khách bị phồng nước (~30m2), tủ bếp bị ẩm mốc, 1 tivi Sony 65 inch bị chập điện do ẩm. Tôi đã chụp ảnh hiện trường."},
            {"speaker": "Bảo hiểm", "text": "Dạ em đã mở hồ sơ. Trong vòng 48h, giám định viên sẽ liên hệ để đến kiểm tra hiện trường. Anh/chị vui lòng giữ nguyên hiện trạng, không tự sửa chữa trước khi giám định. Hồ sơ cần bổ sung: hình ảnh thiệt hại, hóa đơn mua sắm các vật dụng bị hư hỏng (nếu còn), và bản kê khai thiệt hại chi tiết. Thời gian giải quyết dự kiến 7-10 ngày làm việc kể từ khi nhận đủ hồ sơ ạ."},
        ]
    },
    {
        "scenario": "event_planning",
        "turns": [
            {"speaker": "Khách hàng", "text": "Chào bạn, công ty tôi cần tổ chức hội thảo cho khoảng 200 khách vào giữa tháng 9. Bạn tư vấn gói tổ chức sự kiện trọn gói giúp tôi."},
            {"speaker": "Tổ chức sự kiện", "text": "Dạ chào anh/chị. Với quy mô 200 khách, em đề xuất gói Hội Thảo Chuyên Nghiệp bao gồm: thuê hội trường khách sạn 5 sao, hệ thống âm thanh ánh sáng, backdrop + standee, MC song ngữ, teabreak 2 lần, ăn trưa buffet, quà tặng cho khách mời, và chụp ảnh/video sự kiện. Chi phí ước tính khoảng 180-220 triệu tùy địa điểm. Anh/chị thấy thế nào ạ?"},
            {"speaker": "Khách hàng", "text": "Ngân sách của tôi khoảng 150 triệu thôi. Có thể cắt giảm những hạng mục nào để phù hợp?"},
            {"speaker": "Tổ chức sự kiện", "text": "Dạ em đề xuất phương án: chọn khách sạn 4 sao thay vì 5 sao, giảm còn 1 lần teabreak, thay MC song ngữ bằng MC tiếng Việt, và phần quà tặng đơn giản hơn. Chi phí ước tính còn khoảng 145-155 triệu. Hoặc anh/chị có thể chọn gói Hội Thảo Cơ Bản (hội trường 3 sao, setup đơn giản) với chi phí 90-120 triệu."},
        ]
    },
    {
        "scenario": "library_services",
        "turns": [
            {"speaker": "Độc giả", "text": "Chào thư viện, tôi muốn mượn cuốn sách 'Deep Learning với PyTorch' và 'Xử lý ngôn ngữ tự nhiên tiếng Việt'. Không biết thư viện có không ạ?"},
            {"speaker": "Thủ thư", "text": "Dạ chào bạn. Để mình tra cứu trong hệ thống. Cuốn 'Deep Learning với PyTorch' hiện còn 2 bản tại kho A, kệ số 14. Cuốn 'Xử lý ngôn ngữ tự nhiên tiếng Việt' đang có người mượn, dự kiến trả vào ngày 15/07. Bạn có muốn đặt trước không?"},
            {"speaker": "Độc giả", "text": "Vâng, cho tôi đặt trước cuốn thứ hai. Tôi có thể mượn tối đa bao nhiêu cuốn và trong bao lâu?"},
            {"speaker": "Thủ thư", "text": "Sinh viên được mượn tối đa 5 cuốn, thời hạn 30 ngày, có thể gia hạn 1 lần thêm 15 ngày nếu sách không có người đặt trước. Bạn cần mang theo thẻ sinh viên hoặc căn cước công dân để làm thẻ thư viện nếu chưa có. Phí làm thẻ là 50,000đ, có giá trị trong suốt thời gian học."},
        ]
    },
    {
        "scenario": "home_repair_service",
        "turns": [
            {"speaker": "Chủ nhà", "text": "Nhà tôi bị thấm nước ở trần phòng ngủ tầng 2 mỗi khi trời mưa to. Tôi cần thợ đến kiểm tra và sửa chữa gấp."},
            {"speaker": "Dịch vụ sửa chữa", "text": "Dạ chào anh/chị. Vấn đề thấm trần thường do nứt mái hoặc thoát nước bị tắc. Bên em sẽ cử kỹ thuật viên đến khảo sát miễn phí trong ngày mai. Anh/chị cho em xin địa chỉ và số điện thoại để đặt lịch ạ."},
            {"speaker": "Chủ nhà", "text": "Địa chỉ 123 Nguyễn Văn Linh, quận 7. SĐT 0912xxxxxx. Chi phí sửa chữa ước tính bao nhiêu và bảo hành thế nào?"},
            {"speaker": "Dịch vụ sửa chữa", "text": "Dạ em đã ghi nhận. Kỹ thuật viên sẽ đến lúc 9h sáng mai. Về chi phí, sau khi khảo sát sẽ có báo giá chi tiết. Thông thường xử lý thấm trần dao động 2-5 triệu/m2 tùy mức độ hư hỏng và phương pháp xử lý (phun PU hay trải màng chống thấm). Bên em bảo hành 24 tháng cho hạng mục chống thấm. Nếu trong thời gian bảo hành bị thấm lại, sửa miễn phí hoàn toàn ạ."},
        ]
    },
    {
        "scenario": "gym_membership",
        "turns": [
            {"speaker": "Khách hàng", "text": "Tôi muốn tìm hiểu về các gói tập gym bên mình. Tôi thường tập vào buổi sáng trước khi đi làm, chủ yếu tập cardio và yoga."},
            {"speaker": "Tư vấn viên", "text": "Dạ chào anh/chị. Bên em có 3 gói: Basic (990k/tháng, giờ hành chính), Standard (1.5tr/tháng, full ngày + yoga miễn phí), Premium (2.5tr/tháng, full ngày + PT 2 buổi/tuần + xông hơi + bơi). Với nhu cầu của anh/chị, gói Standard là phù hợp nhất. Anh/chị có thể dùng thử 1 buổi miễn phí trước khi đăng ký ạ."},
            {"speaker": "Khách hàng", "text": "OK cho tôi đăng ký buổi dùng thử vào sáng thứ Hai. Tôi cũng muốn biết phòng tập có đông vào buổi sáng không? Tôi không thích chờ đợi máy."},
            {"speaker": "Tư vấn viên", "text": "Buổi sáng từ 6h-8h là khung giờ khá vắng, khoảng 15-20 người, anh/chị sẽ không phải chờ máy. Phòng tập rộng 2000m2 với 80 máy cardio, khu yoga riêng 200m2. Em sẽ đặt lịch dùng thử cho anh/chị 7h sáng thứ Hai. Anh/chị mang theo đồ tập và giày thể thao là được ạ."},
        ]
    },
]

# ============================================================================
# NEEDLE-IN-HAYSTACK
# ============================================================================

NEEDLES = [
    {
        "needle": "Mật khẩu truy cập hệ thống quản lý nội bộ là VIETCOMPRESS2026_SECURE.",
        "context_template": "Theo báo cáo thường niên của Tập đoàn Công nghệ Việt, năm 2026 đánh dấu bước phát triển vượt bậc trong lĩnh vực trí tuệ nhân tạo tại Việt Nam. Các công ty công nghệ hàng đầu đã đầu tư hơn 5,000 tỷ đồng vào nghiên cứu và phát triển AI, tập trung vào ba lĩnh vực chính: xử lý ngôn ngữ tự nhiên, thị giác máy tính, và robotics. Đặc biệt, các mô hình ngôn ngữ lớn như PhoGPT và các phiên bản fine-tune cho tiếng Việt đã đạt được những tiến bộ đáng kể trong các tác vụ như trả lời câu hỏi, tóm tắt văn bản, và sinh nội dung sáng tạo. Hệ thống quản lý nội bộ của tập đoàn yêu cầu xác thực hai lớp, sử dụng mã OTP gửi qua SMS kết hợp với mật khẩu cố định.",
        "query": "Mật khẩu truy cập hệ thống quản lý nội bộ của Tập đoàn Công nghệ Việt là gì?",
        "answer": "VIETCOMPRESS2026_SECURE",
        "insert_position": "middle"
    },
    {
        "needle": "Số tài khoản ngân hàng cần chuyển khoản là 1903666888666 mở tại Ngân hàng Ngoại thương Việt Nam chi nhánh Hà Nội.",
        "context_template": "Hợp đồng mua bán số 2026/HĐMB-TCKT ngày 15 tháng 03 năm 2026 giữa Công ty TNHH Thương mại Dịch vụ Ánh Dương (bên mua) và Công ty Cổ phần Thiết bị Y tế MedTech Việt Nam (bên bán). Điều 1: Đối tượng hợp đồng - Bên bán đồng ý bán và bên mua đồng ý mua 50 máy siêu âm Doppler xách tay model SonoBook 9, xuất xứ Hàn Quốc. Điều 2: Giá trị hợp đồng - Tổng giá trị hợp đồng là 12,500,000,000 đồng (Mười hai tỷ năm trăm triệu đồng chẵn), đã bao gồm thuế VAT 10%. Điều 3: Phương thức thanh toán - Bên mua thanh toán thành 3 đợt.",
        "query": "Số tài khoản ngân hàng cần chuyển khoản trong hợp đồng mua bán thiết bị y tế là gì và mở tại ngân hàng nào?",
        "answer": "1903666888666, Ngân hàng Ngoại thương Việt Nam (Vietcombank) chi nhánh Hà Nội",
        "insert_position": "end"
    },
    {
        "needle": "Ngày họp Đại hội đồng cổ đông thường niên sẽ diễn ra vào lúc 09:00 sáng thứ Sáu ngày 25 tháng 07 năm 2026.",
        "context_template": "Tập đoàn Đầu tư và Phát triển Công nghiệp Việt Nam (VID Group) xin trân trọng thông báo đến toàn thể cổ đông về việc triệu tập cuộc họp quan trọng. Tập đoàn đã trải qua một năm tài chính 2025-2026 với nhiều biến động của thị trường chứng khoán trong nước và quốc tế. Chỉ số VN-Index đã có những phiên giao dịch biến động mạnh, dao động từ 1,150 đến 1,380 điểm. Trong bối cảnh đó, VID Group đã duy trì được tốc độ tăng trưởng ổn định với doanh thu thuần đạt 8,500 tỷ đồng, lợi nhuận sau thuế đạt 1,200 tỷ đồng, tăng 15% so với cùng kỳ năm trước.",
        "query": "Đại hội đồng cổ đông thường niên của VID Group sẽ diễn ra vào ngày nào và lúc mấy giờ?",
        "answer": "09:00 sáng thứ Sáu ngày 25 tháng 07 năm 2026",
        "insert_position": "middle"
    },
    {
        "needle": "Liều dùng khuyến cáo là 500mg, uống 2 lần mỗi ngày sau bữa ăn trong 7 ngày liên tục.",
        "context_template": "Thuốc Amoxicillin 500mg là kháng sinh phổ rộng thuộc nhóm penicillin, được chỉ định trong điều trị các nhiễm khuẩn đường hô hấp trên và dưới, nhiễm khuẩn tai mũi họng, nhiễm khuẩn đường tiết niệu, và nhiễm khuẩn da và mô mềm do các vi khuẩn nhạy cảm gây ra. Chống chỉ định với bệnh nhân có tiền sử dị ứng với penicillin hoặc cephalosporin. Thận trọng khi sử dụng cho bệnh nhân suy thận, phụ nữ có thai và cho con bú.",
        "query": "Liều dùng khuyến cáo của thuốc Amoxicillin 500mg là gì?",
        "answer": "500mg, uống 2 lần mỗi ngày sau bữa ăn trong 7 ngày liên tục",
        "insert_position": "end"
    },
    {
        "needle": "Mức lương khởi điểm cho vị trí Kỹ sư AI là 35,000,000 VND/tháng cùng với gói cổ phiếu thưởng trị giá 200,000,000 VND sau 2 năm.",
        "context_template": "Công ty Cổ phần Giải pháp Trí tuệ Nhân tạo Việt Nam (VAI Solutions) là một trong những công ty khởi nghiệp công nghệ phát triển nhanh nhất tại Việt Nam trong lĩnh vực AI. Thành lập năm 2022, đến nay công ty đã có hơn 200 nhân viên với các văn phòng tại Hà Nội, Đà Nẵng và Thành phố Hồ Chí Minh. Công ty chuyên cung cấp các giải pháp AI cho doanh nghiệp trong lĩnh vực tài chính, bảo hiểm, bán lẻ và y tế. Các sản phẩm chính bao gồm hệ thống phát hiện gian lận giao dịch, chatbot chăm sóc khách hàng thông minh, và nền tảng phân tích dữ liệu lớn sử dụng machine learning.",
        "query": "Mức lương khởi điểm cho vị trí Kỹ sư AI tại VAI Solutions là bao nhiêu?",
        "answer": "35,000,000 VND/tháng cộng gói cổ phiếu thưởng 200,000,000 VND sau 2 năm",
        "insert_position": "middle"
    },
    {
        "needle": "Mã giảm giá đặc biệt SUMMER2026 giảm 40% cho tất cả đơn hàng trên 500,000đ, có hiệu lực đến hết ngày 31/08/2026.",
        "context_template": "Chương trình khuyến mãi mùa hè của hệ thống siêu thị điện máy Miền Nam đã chính thức khởi động từ ngày 01/06/2026. Hàng nghìn sản phẩm điện tử, điện lạnh, gia dụng được giảm giá từ 10% đến 50%. Đặc biệt, khách hàng thân thiết sẽ nhận được thêm ưu đãi khi mua sắm trong thời gian diễn ra chương trình. Hệ thống có 25 chi nhánh trên toàn quốc, phục vụ hơn 1 triệu lượt khách mỗi tháng. Các sản phẩm bán chạy nhất bao gồm tivi Samsung, tủ lạnh LG, máy giặt Panasonic và điều hòa Daikin.",
        "query": "Mã giảm giá đặc biệt mùa hè 2026 của siêu thị điện máy Miền Nam là gì và giảm bao nhiêu phần trăm?",
        "answer": "SUMMER2026 giảm 40%",
        "insert_position": "end"
    },
    {
        "needle": "Địa chỉ IP của máy chủ database production là 192.168.55.107, cổng kết nối 5432, sử dụng PostgreSQL 15.",
        "context_template": "Tài liệu hướng dẫn triển khai hệ thống quản lý khách hàng CRM phiên bản 4.2 trên môi trường production. Hệ thống bao gồm ba thành phần chính: frontend React.js được deploy trên Nginx, backend Python FastAPI chạy trên Gunicorn với 4 workers, và cơ sở dữ liệu quan hệ lưu trữ thông tin khách hàng và giao dịch. Kiến trúc microservices được container hóa bằng Docker và orchestrate bởi Kubernetes trên cụm 3 node. Hệ thống sử dụng Redis làm cache layer và RabbitMQ làm message broker cho các tác vụ bất đồng bộ như gửi email và tạo báo cáo định kỳ.",
        "query": "Địa chỉ IP và cổng kết nối của máy chủ database production là gì?",
        "answer": "192.168.55.107, cổng 5432",
        "insert_position": "middle"
    },
    {
        "needle": "Công thức pha chế signature cocktail 'Hạ Long Sunset' gồm: 45ml rượu vodka, 20ml nước cốt chanh dây, 15ml siro hoa atiso đỏ, 10ml nước cốt chanh, đá viên và trang trí bằng lá bạc hà.",
        "context_template": "Quán bar Sky Lounge tọa lạc trên tầng 36 của tòa nhà Landmark 72, Hà Nội, là một trong những địa điểm ngắm hoàng hôn đẹp nhất thành phố. Với tầm nhìn panorama 360 độ bao quát toàn cảnh Hồ Tây, cầu Nhật Tân và dãy núi Ba Vì phía xa, Sky Lounge thu hút đông đảo du khách và giới trẻ Hà Thành. Quán phục vụ hơn 100 loại cocktail và mocktail được pha chế bởi các bartender chuyên nghiệp từng đoạt giải trong các cuộc thi trong nước và quốc tế. Không gian được thiết kế theo phong cách hiện đại kết hợp với các yếu tố truyền thống Việt Nam như đèn lồng Hội An và gốm Bát Tràng.",
        "query": "Công thức pha chế cocktail 'Hạ Long Sunset' gồm những nguyên liệu gì?",
        "answer": "45ml vodka, 20ml nước cốt chanh dây, 15ml siro atiso đỏ, 10ml nước cốt chanh, đá, lá bạc hà",
        "insert_position": "end"
    },
    {
        "needle": "Chính sách bảo hành mở rộng Diamond Care có giá 4,500,000 VND cho 3 năm, bao gồm thay thế linh kiện miễn phí và vệ sinh máy định kỳ 6 tháng/lần.",
        "context_template": "Trung tâm bảo hành ủy quyền của Apple tại Việt Nam cung cấp dịch vụ sửa chữa và bảo trì cho tất cả các sản phẩm chính hãng bao gồm iPhone, iPad, MacBook, Apple Watch và AirPods. Đội ngũ kỹ thuật viên được Apple chứng nhận trực tiếp, sử dụng linh kiện chính hãng và tuân thủ quy trình sửa chữa chuẩn toàn cầu. Thời gian sửa chữa trung bình từ 2-5 ngày làm việc tùy mức độ hư hỏng. Khách hàng có thể đặt lịch hẹn trực tuyến qua website hoặc ứng dụng Apple Support, hoặc đến trực tiếp 5 trung tâm bảo hành tại Hà Nội, TP.HCM và Đà Nẵng.",
        "query": "Gói bảo hành mở rộng Diamond Care có giá bao nhiêu và bao gồm những dịch vụ gì?",
        "answer": "4,500,000 VND cho 3 năm, thay thế linh kiện miễn phí, vệ sinh định kỳ 6 tháng/lần",
        "insert_position": "middle"
    },
]

# ============================================================================
# AGENT TOOL-CALLING
# ============================================================================

AGENT_TASKS = [
    {
        "task_id": "agent_weather_query",
        "description": "Query weather for multiple cities and compare",
        "context": """Bạn là trợ lý AI hỗ trợ người dùng tra cứu thông tin thời tiết. Bạn có các công cụ sau:
1. get_weather(city: str, date: str) -> dict: Lấy thông tin thời tiết của thành phố vào ngày cụ thể. Trả về dict với các key: temperature, humidity, condition, wind_speed.
2. compare_weather(city1: str, city2: str, date: str) -> dict: So sánh thời tiết giữa hai thành phố.
3. recommend_activity(weather: dict) -> list[str]: Gợi ý hoạt động phù hợp dựa trên thời tiết.

Người dùng yêu cầu: "So sánh thời tiết giữa Hà Nội và Đà Nẵng vào ngày mai, và gợi ý hoạt động phù hợp cho từng thành phố."

Hãy lập kế hoạch hành động và trả lời người dùng.""",
        "expected_tools": ["get_weather", "get_weather", "compare_weather", "recommend_activity"],
        "query": "Hãy lập kế hoạch hành động để so sánh thời tiết Hà Nội và Đà Nẵng ngày mai và gợi ý hoạt động.",
        "reference": "Cần gọi get_weather(Hà Nội), get_weather(Đà Nẵng), sau đó compare_weather và recommend_activity cho từng thành phố."
    },
    {
        "task_id": "agent_calculator_chain",
        "description": "Multi-step calculation with tool calls",
        "context": """Bạn có quyền truy cập vào các công cụ tính toán sau:
1. calculate(expression: str) -> float: Tính giá trị của biểu thức toán học.
2. convert_currency(amount: float, from_currency: str, to_currency: str) -> float: Chuyển đổi tiền tệ.
3. calculate_tax(amount: float, tax_rate: float) -> float: Tính thuế.

Người dùng: "Tôi muốn mua một sản phẩm giá 299 USD từ Mỹ. Tỉ giá USD/VND hiện tại là 25,500. Phí vận chuyển là 15% giá sản phẩm. Thuế nhập khẩu là 10% trên tổng giá trị sản phẩm + vận chuyển. Tính tổng số tiền tôi phải trả bằng VND."

Hãy thực hiện các bước tính toán.""",
        "expected_tools": ["calculate", "calculate", "calculate", "convert_currency"],
        "query": "Tính toán tổng số tiền phải trả bằng VND cho sản phẩm 299 USD với phí vận chuyển 15% và thuế nhập khẩu 10%.",
        "reference": "product_price_vnd = 299 * 25500 = 7,624,500 VND; shipping = 0.15 * 7,624,500 = 1,143,675 VND; subtotal = 8,768,175 VND; tax = 0.10 * 8,768,175 = 876,818 VND; total = 9,644,993 VND"
    },
    {
        "task_id": "agent_search_and_summarize",
        "description": "Search and summarize information",
        "context": """Bạn là trợ lý nghiên cứu với các công cụ:
1. web_search(query: str, num_results: int) -> list[dict]: Tìm kiếm web, trả về danh sách bài viết với title, snippet, url.
2. fetch_article(url: str) -> str: Lấy nội dung đầy đủ của bài viết.
3. summarize(text: str, max_length: int) -> str: Tóm tắt văn bản.

Người dùng: "Tìm hiểu về chính sách thu hút đầu tư nước ngoài mới nhất của Việt Nam năm 2026 và tóm tắt các điểm chính."

Hãy lập kế hoạch và thực hiện.""",
        "expected_tools": ["web_search", "fetch_article", "summarize"],
        "query": "Tìm kiếm và tóm tắt chính sách thu hút đầu tư nước ngoài mới nhất của Việt Nam năm 2026.",
        "reference": "Tìm kiếm chính sách FDI Việt Nam 2026, chọn bài viết liên quan nhất, đọc nội dung, tóm tắt các điểm chính về ưu đãi thuế, thủ tục đầu tư, lĩnh vực khuyến khích."
    },
    {
        "task_id": "agent_data_pipeline",
        "description": "Multi-step data processing pipeline",
        "context": """Bạn là trợ lý phân tích dữ liệu với các công cụ:
1. query_database(sql: str) -> list[dict]: Truy vấn cơ sở dữ liệu và trả về kết quả.
2. analyze_data(data: list[dict], method: str) -> dict: Phân tích thống kê dữ liệu.
3. generate_chart(data: dict, chart_type: str) -> str: Tạo biểu đồ.
4. send_report(email: str, content: str) -> bool: Gửi báo cáo qua email.

Người dùng: "Truy vấn doanh số bán hàng 6 tháng đầu năm 2026 theo từng khu vực, phân tích xu hướng, tạo biểu đồ và gửi báo cáo cho manager@company.com."

Hãy lập kế hoạch và thực hiện.""",
        "expected_tools": ["query_database", "analyze_data", "generate_chart", "send_report"],
        "query": "Phân tích doanh số bán hàng 6 tháng 2026 theo khu vực và gửi báo cáo.",
        "reference": "Truy vấn SQL doanh số theo khu vực, analyze_data() với method='trend', generate_chart() loại 'bar', send_report() tới manager@company.com"
    },
    {
        "task_id": "agent_appointment_scheduler",
        "description": "Multi-step appointment scheduling",
        "context": """Bạn là trợ lý lịch hẹn với các công cụ:
1. check_calendar(person: str, date: str) -> list[str]: Kiểm tra lịch trống của một người vào ngày cụ thể.
2. find_common_slot(persons: list[str], date: str) -> list[str]: Tìm khung giờ chung cho nhiều người.
3. book_meeting(attendees: list[str], datetime: str, duration: int, title: str) -> str: Đặt phòng họp và gửi invitation.
4. send_reminder(meeting_id: str, hours_before: int) -> bool: Gửi nhắc nhở trước giờ họp.

Người dùng: "Đặt cuộc họp review sprint cho team 5 người: anh, chị, manager, tech lead và designer vào thứ Sáu tuần này, thời lượng 1 giờ, và gửi nhắc nhở trước 30 phút."

Hãy lập kế hoạch và thực hiện.""",
        "expected_tools": ["find_common_slot", "book_meeting", "send_reminder"],
        "query": "Đặt cuộc họp sprint review cho 5 người vào thứ Sáu, 1 giờ, nhắc trước 30 phút.",
        "reference": "find_common_slot cho 5 attendee, book_meeting với duration=60, send_reminder với hours_before=0.5"
    },
    {
        "task_id": "agent_ecommerce_order",
        "description": "E-commerce order fulfillment flow",
        "context": """Bạn là trợ lý xử lý đơn hàng với các công cụ:
1. verify_stock(product_id: str, quantity: int) -> dict: Kiểm tra tồn kho.
2. calculate_shipping(address: str, weight: float) -> dict: Tính phí vận chuyển và thời gian giao hàng.
3. create_invoice(order: dict) -> str: Tạo hóa đơn và lấy mã thanh toán.
4. process_payment(invoice_id: str, method: str) -> bool: Xử lý thanh toán.

Người dùng: "Khách hàng Nguyễn Thị Mai đặt 2 chiếc điện thoại Samsung Galaxy S25 Ultra, giao đến 456 Lê Lợi, Quận 1, TP.HCM. Thanh toán bằng thẻ tín dụng."

Hãy lập kế hoạch và thực hiện.""",
        "expected_tools": ["verify_stock", "calculate_shipping", "create_invoice", "process_payment"],
        "query": "Xử lý đơn hàng 2 Samsung S25 Ultra cho Nguyễn Thị Mai, giao Quận 1, thanh toán thẻ tín dụng.",
        "reference": "verify_stock('S25U', 2), calculate_shipping('456 Le Loi, Q1'), create_invoice với order details, process_payment với method='credit_card'"
    },
    {
        "task_id": "agent_content_creation",
        "description": "Content creation pipeline",
        "context": """Bạn là trợ lý sáng tạo nội dung với các công cụ:
1. research_topic(topic: str) -> list[str]: Nghiên cứu và thu thập thông tin về chủ đề.
2. generate_outline(keywords: list[str]) -> str: Tạo dàn ý bài viết.
3. write_section(outline: str, section: int) -> str: Viết một phần của bài viết.
4. optimize_seo(content: str, target_keyword: str) -> str: Tối ưu SEO cho nội dung.
5. publish_article(content: str, platform: str) -> str: Đăng bài lên nền tảng.

Người dùng: "Viết một bài blog 2000 từ về 'Xu hướng AI trong chăm sóc sức khỏe tại Việt Nam 2026', tối ưu SEO với từ khóa 'AI y tế Việt Nam', và đăng lên WordPress."

Hãy lập kế hoạch và thực hiện.""",
        "expected_tools": ["research_topic", "generate_outline", "write_section", "optimize_seo", "publish_article"],
        "query": "Viết blog về AI trong chăm sóc sức khỏe Việt Nam 2026, tối ưu SEO, đăng WordPress.",
        "reference": "research_topic('AI y tế Việt Nam'), generate_outline, write_section, optimize_seo với keyword 'AI y tế Việt Nam', publish_article lên WordPress"
    },
    {
        "task_id": "agent_code_review",
        "description": "Automated code review pipeline",
        "context": """Bạn là trợ lý review code với các công cụ:
1. fetch_pr(repo: str, pr_number: int) -> dict: Lấy thông tin Pull Request.
2. analyze_diff(diff: str) -> list[dict]: Phân tích code changes.
3. run_tests(repo: str, branch: str) -> dict: Chạy bộ test tự động.
4. post_review(pr_number: int, comments: list[str], approve: bool) -> bool: Đăng review lên PR.

Người dùng: "Review PR #342 trên repo vncompress/main, phân tích code changes, chạy tests, và nếu tests pass thì approve."

Hãy lập kế hoạch và thực hiện.""",
        "expected_tools": ["fetch_pr", "analyze_diff", "run_tests", "post_review"],
        "query": "Review PR #342 repo vncompress, chạy tests, auto-approve nếu pass.",
        "reference": "fetch_pr('vncompress', 342), analyze_diff, run_tests, post_review với approve=True nếu tests pass"
    },
]

# ============================================================================
# CROSS-LINGUAL
# ============================================================================

CROSS_LINGUAL_SAMPLES = [
    {
        "id": "cross_economics",
        "text_vi": "Kinh tế Việt Nam năm 2026 dự kiến tăng trưởng GDP đạt 6.8%, tiếp tục là một trong những nền kinh tế tăng trưởng nhanh nhất khu vực Đông Nam Á. Các động lực tăng trưởng chính bao gồm xuất khẩu hàng hóa công nghệ cao, đầu tư trực tiếp nước ngoài (FDI) vào lĩnh vực sản xuất chip bán dẫn và năng lượng tái tạo, cùng với sự phục hồi mạnh mẽ của ngành du lịch và dịch vụ. Lạm phát được kiểm soát ở mức 3.5%, trong khi tỉ lệ thất nghiệp giảm xuống còn 2.1%. Chính phủ đã ban hành nhiều chính sách cải cách hành chính, cắt giảm thủ tục đầu tư và nâng cao tính minh bạch của môi trường kinh doanh. Tuy nhiên, thách thức vẫn còn ở chất lượng nguồn nhân lực, cơ sở hạ tầng logistics và khả năng thích ứng với biến đổi khí hậu.",
        "text_en": "Vietnam's economy in 2026 is projected to achieve GDP growth of 6.8%, continuing to be one of the fastest-growing economies in Southeast Asia. Key growth drivers include exports of high-tech goods, foreign direct investment (FDI) in semiconductor manufacturing and renewable energy, along with a strong recovery in tourism and services. Inflation is controlled at 3.5%, while the unemployment rate has decreased to 2.1%. The government has issued various administrative reform policies, streamlined investment procedures, and enhanced the transparency of the business environment. However, challenges remain in workforce quality, logistics infrastructure, and climate change adaptation capacity.",
        "domain": "economics"
    },
    {
        "id": "cross_environment",
        "text_vi": "Việt Nam là một trong những quốc gia chịu ảnh hưởng nặng nề nhất bởi biến đổi khí hậu, đặc biệt là hiện tượng nước biển dâng tại khu vực Đồng bằng sông Cửu Long. Theo ước tính của Ngân hàng Thế giới, nếu mực nước biển dâng thêm 1 mét, khoảng 40% diện tích Đồng bằng sông Cửu Long sẽ bị ngập vĩnh viễn, ảnh hưởng trực tiếp đến sinh kế của hơn 20 triệu người dân. Chính phủ Việt Nam đã cam kết đạt mức phát thải ròng bằng 0 vào năm 2050 tại COP26 và đang triển khai nhiều dự án năng lượng tái tạo quy mô lớn, bao gồm điện gió ngoài khơi và điện mặt trời tập trung.",
        "text_en": "Vietnam is one of the countries most severely affected by climate change, particularly by sea level rise in the Mekong Delta region. According to World Bank estimates, if sea levels rise by 1 meter, approximately 40% of the Mekong Delta area will be permanently flooded, directly affecting the livelihoods of over 20 million people. The Vietnamese government has committed to achieving net-zero emissions by 2050 at COP26 and is implementing numerous large-scale renewable energy projects, including offshore wind power and concentrated solar power.",
        "domain": "environment"
    },
    {
        "id": "cross_ai_technology",
        "text_vi": "Trí tuệ nhân tạo (AI) đang trở thành một trong những lĩnh vực đầu tư trọng điểm của Việt Nam trong chiến lược chuyển đổi số quốc gia. Theo báo cáo của Bộ Thông tin và Truyền thông, đến năm 2026, Việt Nam đã có hơn 200 doanh nghiệp khởi nghiệp trong lĩnh vực AI, tập trung vào các ứng dụng như xử lý ngôn ngữ tự nhiên tiếng Việt, thị giác máy tính trong sản xuất công nghiệp, và phân tích dữ liệu lớn trong tài chính - ngân hàng. Các trường đại học lớn như Đại học Bách Khoa Hà Nội, Đại học Quốc gia TP.HCM đã mở các chương trình đào tạo chuyên sâu về AI và Khoa học dữ liệu.",
        "text_en": "Artificial Intelligence (AI) is becoming one of Vietnam's key investment areas in the national digital transformation strategy. According to a report by the Ministry of Information and Communications, by 2026, Vietnam had over 200 AI startups, focusing on applications such as Vietnamese natural language processing, computer vision in industrial manufacturing, and big data analytics in finance and banking. Major universities such as Hanoi University of Science and Technology and Vietnam National University Ho Chi Minh City have launched specialized programs in AI and Data Science.",
        "domain": "technology"
    },
    {
        "id": "cross_education",
        "text_vi": "Hệ thống giáo dục Việt Nam đang trải qua quá trình đổi mới toàn diện với trọng tâm là chuyển đổi số và phát triển kỹ năng thế kỷ 21 cho học sinh. Bộ Giáo dục và Đào tạo đã triển khai chương trình giáo dục phổ thông mới từ năm 2020, chú trọng phát triển năng lực và phẩm chất người học thay vì truyền thụ kiến thức một chiều. Đến năm 2026, hơn 90% trường học trên toàn quốc đã được kết nối internet, tạo điều kiện cho việc ứng dụng công nghệ thông tin trong dạy và học. Các mô hình giáo dục STEM, STEAM được triển khai rộng rãi từ cấp tiểu học đến trung học phổ thông.",
        "text_en": "Vietnam's education system is undergoing comprehensive reform with a focus on digital transformation and developing 21st century skills for students. The Ministry of Education and Training has implemented the new general education curriculum since 2020, emphasizing the development of learner competencies and qualities rather than one-way knowledge transmission. By 2026, over 90% of schools nationwide have been connected to the internet, enabling the application of information technology in teaching and learning. STEM and STEAM education models have been widely deployed from primary to high school levels.",
        "domain": "education"
    },
    {
        "id": "cross_healthcare",
        "text_vi": "Ngành y tế Việt Nam đã đạt được nhiều thành tựu quan trọng trong việc kiểm soát dịch bệnh và nâng cao chất lượng chăm sóc sức khỏe nhân dân. Tỉ lệ bao phủ bảo hiểm y tế đạt 93% dân số vào năm 2026, tiến gần đến mục tiêu bảo hiểm y tế toàn dân. Việt Nam cũng đã làm chủ nhiều kỹ thuật y học tiên tiến như ghép tạng, phẫu thuật nội soi robot, và điều trị ung thư bằng liệu pháp miễn dịch. Ứng dụng trí tuệ nhân tạo trong chẩn đoán hình ảnh và phân tích dữ liệu y tế đang được triển khai tại các bệnh viện lớn.",
        "text_en": "Vietnam's healthcare sector has achieved many important accomplishments in disease control and improving the quality of public healthcare. Health insurance coverage reached 93% of the population by 2026, approaching the goal of universal health coverage. Vietnam has also mastered many advanced medical techniques such as organ transplantation, robotic endoscopic surgery, and cancer treatment with immunotherapy. The application of artificial intelligence in medical imaging diagnosis and health data analysis is being deployed at major hospitals.",
        "domain": "healthcare"
    },
    {
        "id": "cross_transportation",
        "text_vi": "Hệ thống giao thông Việt Nam đang được đầu tư nâng cấp mạnh mẽ với nhiều dự án hạ tầng quy mô lớn. Tuyến đường sắt đô thị Cát Linh - Hà Đông đã đi vào hoạt động, phục vụ hơn 15,000 lượt khách mỗi ngày. Tuyến metro số 1 Bến Thành - Suối Tiên tại TP.HCM dự kiến hoàn thành vào cuối năm 2026. Dự án đường bộ cao tốc Bắc - Nam dài 2,063 km đang được đẩy nhanh tiến độ, phấn đấu hoàn thành vào năm 2030. Ngành hàng không cũng phát triển nhanh chóng với sân bay quốc tế Long Thành giai đoạn 1 dự kiến đưa vào khai thác năm 2026.",
        "text_en": "Vietnam's transportation system is being heavily invested in and upgraded with many large-scale infrastructure projects. The Cat Linh - Ha Dong urban railway has been operational, serving over 15,000 passengers daily. Metro line 1 Ben Thanh - Suoi Tien in Ho Chi Minh City is expected to be completed by the end of 2026. The North-South expressway project spanning 2,063 km is being accelerated, aiming for completion by 2030. The aviation sector is also developing rapidly with Long Thanh International Airport phase 1 expected to begin operations in 2026.",
        "domain": "transportation"
    },
]

# ============================================================================
# DATASET BUILDER
# ============================================================================

def load_wikipedia_data() -> List[Dict]:
    """Load pre-fetched Wikipedia paragraphs."""
    path = os.path.join(DATA_DIR, 'wikipedia_vi_raw.json')
    if not os.path.exists(path):
        print(f"[WARN] Wikipedia data not found at {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('paragraphs', [])


def build_paragraphs_by_length(
    paragraphs: List[Dict],
    num_bins: int = 4
) -> Dict[str, List[Dict]]:
    """Group paragraphs by character length for stratified sampling."""
    lengths = [p['char_length'] for p in paragraphs]
    if not lengths:
        return {}
    min_l, max_l = min(lengths), max(lengths)
    bin_size = (max_l - min_l) / num_bins

    bins = {}
    for p in paragraphs:
        l = p['char_length']
        bin_idx = min(int((l - min_l) / bin_size), num_bins - 1)
        bin_key = f"len_{int(min_l + bin_idx * bin_size)}_{int(min_l + (bin_idx + 1) * bin_size)}"
        bins.setdefault(bin_key, []).append(p)
    return bins


def build_long_document_qa(
    paragraphs: List[Dict],
    num_paragraphs: int = 50,
    queries_per_para: int = 3
) -> List[Dict]:
    """Build Long-Document QA task from Wikipedia paragraphs.

    Each paragraph gets multiple query variants for diversity.
    """
    samples = []
    bins = build_paragraphs_by_length(paragraphs)
    all_paras = [p for b in bins.values() for p in b] if bins else paragraphs

    if len(all_paras) > num_paragraphs:
        selected = random.sample(all_paras, num_paragraphs)
    else:
        selected = all_paras

    query_templates = [
        "Bài viết về {title} đề cập đến những nội dung chính nào?",
        "Hãy tóm tắt những điểm quan trọng nhất trong đoạn văn về {title}.",
        "Những thông tin chính được đề cập trong bài viết về {title} là gì?",
    ]

    for i, para in enumerate(selected):
        text = para['text']
        title = para.get('title', 'chủ đề này')
        for qi, q_template in enumerate(query_templates[:queries_per_para]):
            query = q_template.format(title=title)
            samples.append({
                'sample_id': f"doc_qa_{i:04d}_q{qi}",
                'task': 'long_document_qa',
                'title': title,
                'context': text,
                'query': query,
                'reference_answer': text,
                'domain': para.get('topic_id', 'general'),
                'source': para.get('source', 'wikipedia'),
                'char_length': para['char_length'],
            })

    return samples


def build_multi_turn_conversation(num_samples: int = 20) -> List[Dict]:
    """Build Multi-turn Conversation Summarization task."""
    samples = []

    for i in range(min(num_samples, len(VIETNAMESE_CONVERSATIONS))):
        conv = VIETNAMESE_CONVERSATIONS[i]
        full_text = '\n'.join(
            [f"{t['speaker']}: {t['text']}" for t in conv['turns']]
        )

        queries = [
            "Hãy tóm tắt cuộc hội thoại trên.",
            f"Cuộc hội thoại về {conv['scenario']} này có những nội dung chính gì?",
            "Tóm tắt những điểm quan trọng được thảo luận trong cuộc trò chuyện.",
        ]
        query = random.choice(queries)

        reference = ' '.join([t['text'] for t in conv['turns']])

        samples.append({
            'sample_id': f"conv_{i:04d}",
            'task': 'multi_turn_conversation',
            'scenario': conv['scenario'],
            'context': full_text,
            'query': query,
            'reference_answer': reference,
            'num_turns': len(conv['turns']),
            'char_length': len(full_text),
        })

    return samples


def build_legal_document_qa(num_samples: int = 15) -> List[Dict]:
    """Build legal document QA from built-in legal texts."""
    samples = []
    sample_idx = 0

    for law_id, law_data in VIETNAMESE_LEGAL_TEXTS.items():
        for chapter_name, articles in law_data['chapters'].items():
            full_text = '\n'.join(articles)
            if len(full_text) < 100:
                continue

            queries = [
                f"Theo {law_data['title']}, {chapter_name} quy định những nội dung gì?",
                f"Trích dẫn các quy định chính trong {chapter_name} của {law_data['title']}.",
                f"Những điều khoản quan trọng nào được đề cập trong {chapter_name}?",
            ]
            query = random.choice(queries)

            samples.append({
                'sample_id': f"legal_{sample_idx:04d}",
                'task': 'long_document_qa',
                'law_id': law_id,
                'law_title': law_data['title'],
                'chapter': chapter_name,
                'context': full_text,
                'query': query,
                'reference_answer': full_text,
                'domain': 'legal',
                'char_length': len(full_text),
            })
            sample_idx += 1

    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)

    return samples


def build_needle_in_haystack(
    paragraphs: List[Dict],
    num_distractors: int = 3
) -> List[Dict]:
    """Build Needle-in-Haystack task."""
    samples = []
    bins = build_paragraphs_by_length(paragraphs)

    for i, needle_data in enumerate(NEEDLES):
        distractor_paragraphs = []
        if bins:
            for b_key, b_paragraphs in bins.items():
                if len(b_paragraphs) >= num_distractors:
                    selected = random.sample(b_paragraphs, num_distractors)
                    distractor_paragraphs.extend(selected)

        if not distractor_paragraphs:
            distractor_paragraphs = random.sample(
                paragraphs, min(num_distractors * 3, len(paragraphs))
            )

        context_parts = []
        for p in distractor_paragraphs:
            context_parts.append(p['text'])

        insert_pos = needle_data.get('insert_position', 'middle')
        needle_text = needle_data['needle']

        if insert_pos == 'beginning':
            context_parts.insert(0, needle_text)
        elif insert_pos == 'end':
            context_parts.append(needle_text)
        else:
            mid = len(context_parts) // 2
            context_parts.insert(mid, needle_text)

        full_context = '\n\n'.join(context_parts)

        samples.append({
            'sample_id': f"needle_{i:04d}",
            'task': 'needle_in_haystack',
            'context': full_context,
            'query': needle_data['query'],
            'reference_answer': needle_data['answer'],
            'needle': needle_data['needle'],
            'insert_position': insert_pos,
            'char_length': len(full_context),
            'num_paragraphs': len(context_parts),
        })

    return samples


def build_agent_tool_calling() -> List[Dict]:
    """Build Agent Tool-Calling task."""
    samples = []
    for i, task in enumerate(AGENT_TASKS):
        samples.append({
            'sample_id': f"agent_{i:04d}",
            'task': 'agent_tool_calling',
            'agent_task_id': task['task_id'],
            'description': task['description'],
            'context': task['context'],
            'query': task['query'],
            'reference_answer': task['reference'],
            'expected_tools': task.get('expected_tools', []),
            'char_length': len(task['context']),
        })
    return samples


def build_cross_lingual() -> List[Dict]:
    """Build Cross-lingual Compression task."""
    samples = []
    for i, sample in enumerate(CROSS_LINGUAL_SAMPLES):
        for lang in ['vi_to_vi', 'vi_to_en', 'en_to_en']:
            if lang == 'vi_to_vi':
                text = sample['text_vi']
                query = f"Dịa sang tiếng Anh: {sample['text_vi'][:100]}..."
                ref = sample['text_en']
            elif lang == 'vi_to_en':
                text = sample['text_vi']
                query = "Summarize in English: " + sample['text_en'][:100]
                ref = sample['text_en']
            else:
                text = sample['text_en']
                query = "Summarize this text: " + sample['text_en'][:100]
                ref = sample['text_en']

            samples.append({
                'sample_id': f"cross_{i:04d}_{lang}",
                'task': 'cross_lingual',
                'cross_config': lang,
                'domain': sample['domain'],
                'context': text,
                'vi_text': sample['text_vi'],
                'en_text': sample['text_en'],
                'query': query,
                'reference_answer': ref,
                'char_length': len(text),
            })

    return samples


def validate_dataset(samples: List[Dict]) -> Dict:
    """Validate dataset quality and return statistics."""
    stats = {
        'total_samples': len(samples),
        'task_distribution': {},
        'length_stats': {},
        'issues': [],
    }

    for sample in samples:
        task = sample['task']
        stats['task_distribution'][task] = stats['task_distribution'].get(task, 0) + 1

        cl = sample.get('char_length', len(sample.get('context', '')))
        if task not in stats['length_stats']:
            stats['length_stats'][task] = {'lengths': [], 'min': 0, 'max': 0, 'avg': 0}
        stats['length_stats'][task]['lengths'].append(cl)

        if cl < 200:
            stats['issues'].append(f"Sample {sample.get('sample_id', '?')} too short ({cl} chars)")
        if cl > 50000:
            stats['issues'].append(f"Sample {sample.get('sample_id', '?')} very long ({cl} chars)")
        if not sample.get('context', '').strip():
            stats['issues'].append(f"Sample {sample.get('sample_id', '?')} empty context")
        if not sample.get('query', '').strip():
            stats['issues'].append(f"Sample {sample.get('sample_id', '?')} empty query")

    for task, data in stats['length_stats'].items():
        lengths = data['lengths']
        data['min'] = min(lengths) if lengths else 0
        data['max'] = max(lengths) if lengths else 0
        data['avg'] = int(sum(lengths) / len(lengths)) if lengths else 0
        data['median'] = int(sorted(lengths)[len(lengths)//2]) if lengths else 0
        del data['lengths']

    return stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    print('=' * 60)
    print('VCC-Bench Dataset Builder')
    print('=' * 60)

    print('\n[1/6] Loading Wikipedia data...')
    wiki_paragraphs = load_wikipedia_data()
    print(f'  Loaded {len(wiki_paragraphs)} paragraphs')

    print('[2/6] Building Long-Document QA...')
    doc_qa_samples = build_long_document_qa(wiki_paragraphs, num_paragraphs=50, queries_per_para=3)
    print(f'  {len(doc_qa_samples)} samples ({len(set(s["sample_id"].rsplit("_q",1)[0] for s in doc_qa_samples))} docs x up to 3 queries)')

    print('[3/6] Building Legal Document QA...')
    legal_samples = build_legal_document_qa(num_samples=30)
    print(f'  {len(legal_samples)} samples')

    print('[4/6] Building Multi-turn Conversation Summarization...')
    conv_samples = build_multi_turn_conversation(num_samples=99)
    conv_duplicated = []
    for c in conv_samples:
        for qi in range(3):
            variant = c.copy()
            q_templates = [
                "Hãy tóm tắt cuộc hội thoại trên.",
                f"Cuộc hội thoại về {c['scenario']} này có những nội dung chính gì?",
                "Tóm tắt những điểm quan trọng được thảo luận trong cuộc trò chuyện.",
            ]
            variant['sample_id'] = f"{c['sample_id']}_q{qi}"
            variant['query'] = q_templates[qi]
            conv_duplicated.append(variant)
    conv_samples = conv_duplicated
    print(f'  {len(conv_samples)} samples ({len(VIETNAMESE_CONVERSATIONS)} scenarios x 3 queries)')

    print('[5/6] Building Needle-in-Haystack...')
    needle_samples = build_needle_in_haystack(wiki_paragraphs)
    print(f'  {len(needle_samples)} samples')

    print('[6/6] Building Agent Tool-Calling & Cross-lingual...')
    agent_samples = build_agent_tool_calling()
    cross_samples = build_cross_lingual()
    print(f'  {len(agent_samples)} agent samples, {len(cross_samples)} cross-lingual samples')

    all_samples = (
        doc_qa_samples + legal_samples + conv_samples +
        needle_samples + agent_samples + cross_samples
    )

    print('\n' + '=' * 60)
    print('Dataset Validation')
    print('=' * 60)
    stats = validate_dataset(all_samples)
    print(f'  Total samples: {stats["total_samples"]}')

    task_names = {
        'long_document_qa': 'Long-Doc QA',
        'multi_turn_conversation': 'Multi-turn Conv',
        'needle_in_haystack': 'Needle-in-Haystack',
        'agent_tool_calling': 'Agent Tool-Calling',
        'cross_lingual': 'Cross-lingual',
    }

    print(f'\n  {"Task":<25s} {"Count":>6s} {"Min":>8s} {"Avg":>8s} {"Max":>8s} {"Median":>8s}')
    print(f'  {"-"*25} {"-"*6} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')
    for task, count in sorted(stats['task_distribution'].items()):
        ls = stats['length_stats'].get(task, {})
        name = task_names.get(task, task)
        print(f'  {name:<25s} {count:>6d} {ls.get("min",0):>8d} {ls.get("avg",0):>8d} {ls.get("max",0):>8d} {ls.get("median",0):>8d}')

    if stats['issues']:
        print(f'\n  Issues found ({len(stats["issues"])}):')
        for issue in stats['issues'][:10]:
            print(f'    - {issue}')
        if len(stats['issues']) > 10:
            print(f'    ... and {len(stats["issues"]) - 10} more')
    else:
        print('\n  No quality issues found!')

    dataset = {
        'metadata': {
            'name': 'VCC-Bench v1.0',
            'version': '1.0.0',
            'date': time.strftime('%Y-%m-%d'),
            'language': 'vi',
            'description': 'Vietnamese Context Compression Benchmark',
            'tasks': sorted(list(stats['task_distribution'].keys())),
            'total_samples': len(all_samples),
            'license': 'CC-BY-SA 4.0 (Wikipedia) + Public Domain (Legal) + MIT (synthetic)',
        },
        'statistics': stats,
        'samples': all_samples,
    }

    out_path = os.path.join(DATA_DIR, 'vcc_bench_v1.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    file_size = os.path.getsize(out_path)
    print(f'\nSaved: {out_path}')
    print(f'Size: {file_size / 1024:.1f} KB ({file_size / 1024 / 1024:.2f} MB)')

    # Save per-task files for easier loading
    for task in stats['task_distribution']:
        task_samples = [s for s in all_samples if s['task'] == task]
        task_path = os.path.join(DATA_DIR, f'vcc_bench_{task}.json')
        with open(task_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {**dataset['metadata'], 'task': task, 'num_samples': len(task_samples)},
                'samples': task_samples,
            }, f, ensure_ascii=False, indent=2)
        print(f'  Task file: {task_path} ({len(task_samples)} samples, {os.path.getsize(task_path)/1024:.1f} KB)')

    print(f'\nDone! Created {len(stats["task_distribution"])} task files + 1 combined file.')


if __name__ == '__main__':
    main()
