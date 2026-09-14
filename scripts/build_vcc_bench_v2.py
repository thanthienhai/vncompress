#!/usr/bin/env python3
"""Build VCC-Bench v2 -- fixes the needle-in-haystack confound in v1.

See docs/build_VCC-Bench v2.md for the full spec. Short version of the bug:
v1's needle set (`vcc_bench_needle_in_haystack.json`) had 9 samples, and every
needle payload was a pure alphanumeric string (activation codes, IPs, dates,
prices) -- tone density rho(t)=0 for all of them under the w_tone formula in
`vncompress/linguistics.py`. Wave 1's "2.3% needle recall at 2x" therefore
measured "does compression drop undiacriticized tokens" (yes, at floor
priority, deterministically) and NOT "is tone-aware compression bad for
retrieval in general" -- there was no diacritic-bearing needle to compare
against, so the two explanations were confounded.

v2 fixes this with a controlled needle design:

  needle_in_haystack (120 samples, up from 9):
    - 40 haystacks (`haystack_id`), each built once from wikipedia_vi_raw.json
      paragraphs and shared verbatim across three needle groups -- only the
      needle sentence changes between them (see `build_base_haystack`).
    - Group A (no diacritics): codes/IPs/dates/amounts -- reproduces the v1
      condition.
    - Group B (has diacritics): Vietnamese person names / place names /
      idiomatic phrases -- the missing control group. Payloads are
      independently authored Vietnamese content, NOT group-A strings with
      diacritics stripped back on (that would just produce nonsense tokens
      the generator treats differently for reasons unrelated to tone).
    - Group C (mixed): one sentence carrying both a no-diacritic fact (a
      code/number) and a diacritic-bearing fact (a name/place), to see
      whether the two interact.
    - Insert position (beginning/middle/end) is fixed per haystack_id and
      therefore identical across a sample's A/B/C triple -- "change only the
      needle, hold everything else fixed" is the point of a controlled study.
    - Every payload's tone density is checked with the actual
      `VietnameseToneAnalyzer.compute_tone_density` used by w_tone, not just
      assumed from the category, so `needle_has_diacritic` is measured, not
      guessed.

  long_document_qa (some 220 samples, up from 160):
    - The original 160 v1 samples are carried over byte-for-byte (loaded from
      the frozen `vcc_bench_v1.json`, not regenerated -- regenerating through
      the v1 template functions would depend on matching global `random` call
      order exactly, which is fragile; loading the frozen output is exact and
      trivially correct). Tagged `qa_subset='standard'`.
    - `qa_subset='multi_hop'` (~30): context is a full Wikipedia article
      (all its paragraphs, in order); the query points at two non-adjacent
      paragraphs by their opening words and asks for the information that
      connects them; `evidence_paragraph_indices` records which two so a
      later pass can check whether both survived compression.
    - `qa_subset='referential'` (~30): context is again a full article; one
      evidence paragraph is the article's own introduction (which names the
      subject) and the other is a later paragraph that only refers back to
      that subject via a pronoun/back-reference marker ("Ông/Bà ...", "Điều
      này/đó", "Do đó", "Vì vậy", ...) rather than repeating its name --
      compressing away the introduction breaks the referent even though the
      later paragraph is untouched.
    multi_turn_conversation / agent_tool_calling / cross_lingual are carried
    over unchanged from v1 (loaded, not regenerated) -- the doc's Việc 1/2/3
    only concern needle-in-haystack and long_document_qa.

Determinism: every random draw in this script goes through a single
`random.Random(SEED)` instance (see SEED below), never the global `random`
module -- so re-running this script reproduces the exact same v2 file, and
doesn't perturb/depend on any other script's use of `random`.

Usage:
    python scripts/build_vcc_bench_v2.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from random import Random
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_vcc_bench import DATA_DIR, load_wikipedia_data, validate_dataset  # noqa: E402
from vncompress.linguistics import get_tone_analyzer  # noqa: E402

# Seed = the date this v2 dataset was authored (see PROVENANCE.md "Needle v2
# generation"), kept separate from build_vcc_bench.py's random.seed(42) so
# this script's output never depends on import order or on anything else
# having consumed the global `random` stream first.
SEED = 20260914

NUM_HAYSTACKS = 40
TARGET_CONTEXT_LEN = 30_000  # chars; same target for all 3 groups x 40 haystacks
INSERT_POSITIONS = ['beginning', 'middle', 'end']

TONE = get_tone_analyzer()


def has_diacritic(text: str) -> bool:
    """True iff at least one character carries a Vietnamese tone mark --
    i.e. rho(text) > 0 under the exact w_tone formula this benchmark exists
    to stress-test (vncompress.linguistics.VietnameseToneAnalyzer)."""
    return TONE.compute_tone_density(text) > 0.0


# ============================================================================
# Group A -- no-diacritic payloads (codes, numbers, IPs, dates)
# ============================================================================

def _digits(rng: Random, n: int) -> str:
    return ''.join(rng.choice('0123456789') for _ in range(n))


def _alnum(rng: Random, n: int) -> str:
    # Excludes visually-ambiguous chars (0/O, 1/I) -- cosmetic, not required,
    # but keeps generated codes readable if anyone inspects the file by eye.
    return ''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(n))


def _gen_activation_code(rng):
    code = f"VNC-{rng.randint(2026, 2029)}-{_alnum(rng, 4)}"
    return (f"Mã kích hoạt sản phẩm là {code}.",
            "Mã kích hoạt sản phẩm được nhắc đến trong đoạn văn là gì?", code)


def _gen_ip_port(rng):
    ip = f"{rng.randint(10, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    port = rng.choice([443, 3306, 5432, 6379, 8080, 8443, 9200, 27017])
    payload = f"{ip}:{port}"
    return (f"Địa chỉ IP và cổng của máy chủ backup là {ip}:{port}.",
            "Địa chỉ IP và cổng của máy chủ backup được nhắc đến là gì?", payload)


def _gen_bank_account(rng):
    acct = _digits(rng, rng.choice([10, 12, 13]))
    return (f"Số tài khoản cần đối soát là {acct}.",
            "Số tài khoản cần đối soát trong đoạn văn là gì?", acct)


def _gen_order_id(rng):
    oid = f"DH{rng.randint(202601, 202612)}{_digits(rng, 4)}"
    return (f"Mã đơn hàng cần tra cứu là {oid}.",
            "Mã đơn hàng cần tra cứu được nhắc đến trong đoạn văn là gì?", oid)


def _gen_tracking(rng):
    tn = f"{_alnum(rng, 2)}{_digits(rng, 9)}VN"
    return (f"Mã vận đơn của lô hàng là {tn}.",
            "Mã vận đơn của lô hàng được nhắc đến là gì?", tn)


def _gen_phone(rng):
    ph = f"09{rng.randint(10, 99)}{_digits(rng, 6)}"
    return (f"Số điện thoại hỗ trợ khẩn cấp là {ph}.",
            "Số điện thoại hỗ trợ khẩn cấp được nhắc đến trong đoạn văn là gì?", ph)


def _gen_license_plate(rng):
    plate = f"{rng.randint(11, 99)}{rng.choice('ABCDEFGHKLMNPSTUVXYZ')}-{_digits(rng, 3)}.{_digits(rng, 2)}"
    return (f"Biển số xe được ghi nhận tại hiện trường là {plate}.",
            "Biển số xe được ghi nhận tại hiện trường là gì?", plate)


def _gen_invoice(rng):
    inv = f"HD{_digits(rng, 2)}{_alnum(rng, 3)}{_digits(rng, 4)}"
    return (f"Số hóa đơn điện tử là {inv}.",
            "Số hóa đơn điện tử được nhắc đến trong đoạn văn là gì?", inv)


def _gen_employee_id(rng):
    eid = f"NV{_digits(rng, 6)}"
    return (f"Mã số nhân viên cần tra cứu lương là {eid}.",
            "Mã số nhân viên cần tra cứu lương được nhắc đến là gì?", eid)


def _gen_datetime(rng):
    t = f"{rng.randint(0, 23):02d}:{rng.choice([0, 15, 30, 45]):02d}"
    d = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2026, 2027)}"
    payload = f"{t}, {d}"  # no diacritic connector word ("ngày" carries a huyen tone mark)
    return (f"Giờ khởi hành được xác nhận là {t} ngày {d}.",
            "Giờ khởi hành được xác nhận trong đoạn văn là khi nào?", payload)


def _gen_price(rng):
    amt = f"{rng.randint(1, 999)},{rng.randint(0, 9)}{rng.randint(0, 9)}0,000"
    payload = f"{amt} VND"
    return (f"Tổng chi phí quyết toán là {amt} VND.",
            "Tổng chi phí quyết toán được nhắc đến trong đoạn văn là bao nhiêu?", payload)


def _gen_discount(rng):
    code2 = f"SALE{rng.randint(10, 99)}{_alnum(rng, 2)}"
    pct = rng.choice([10, 15, 20, 25, 30, 40, 50])
    payload = f"{code2}, {pct}%"
    return (f"Mã giảm giá {code2} áp dụng mức {pct}%.",
            "Mã giảm giá và mức giảm được nhắc đến trong đoạn văn là gì?", payload)


def _gen_zip(rng):
    z = _digits(rng, 6)
    return (f"Mã bưu chính của kho hàng là {z}.",
            "Mã bưu chính của kho hàng được nhắc đến là gì?", z)


def _gen_serial(rng):
    sn = f"SN{_alnum(rng, 3)}{_digits(rng, 6)}"
    return (f"Số serial thiết bị cần bảo hành là {sn}.",
            "Số serial thiết bị cần bảo hành được nhắc đến trong đoạn văn là gì?", sn)


def _gen_apikey(rng):
    ak = f"sk_{_alnum(rng, 4)}{_digits(rng, 4)}{_alnum(rng, 4)}"
    return (f"API key cấp cho đối tác là {ak}.",
            "API key cấp cho đối tác được nhắc đến trong đoạn văn là gì?", ak)


def _gen_version(rng):
    v = f"{rng.randint(1, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 20)}"
    return (f"Phiên bản phần mềm đang triển khai là {v}.",
            "Phiên bản phần mềm đang triển khai được nhắc đến là gì?", v)


def _gen_temp(rng):
    temp = rng.randint(-5, 45)
    payload = f"{temp}°C"  # degree symbol, not the Vietnamese word "do" -- keeps payload diacritic-free
    return (f"Nhiệt độ đo được tại thời điểm kiểm tra là {payload}.",
            "Nhiệt độ đo được tại thời điểm kiểm tra được nhắc đến là bao nhiêu?", payload)


def _gen_coordinate(rng):
    lat = f"{rng.randint(8, 23)}.{_digits(rng, 4)}"
    lon = f"{rng.randint(102, 109)}.{_digits(rng, 4)}"
    payload = f"{lat}, {lon}"
    return (f"Tọa độ GPS ghi nhận được là {lat}, {lon}.",
            "Tọa độ GPS ghi nhận được trong đoạn văn là gì?", payload)


def _gen_hexcolor(rng):
    hexcolor = "#" + ''.join(rng.choice('0123456789ABCDEF') for _ in range(6))
    return (f"Mã màu thương hiệu chính thức là {hexcolor}.",
            "Mã màu thương hiệu chính thức được nhắc đến trong đoạn văn là gì?", hexcolor)


def _gen_flight(rng):
    fl = f"VN{_digits(rng, 3)}"
    return (f"Số hiệu chuyến bay bị hoãn là {fl}.",
            "Số hiệu chuyến bay bị hoãn được nhắc đến trong đoạn văn là gì?", fl)


GROUP_A_GENERATORS = [
    ('activation_code', _gen_activation_code),
    ('ip_port', _gen_ip_port),
    ('bank_account', _gen_bank_account),
    ('order_id', _gen_order_id),
    ('tracking_number', _gen_tracking),
    ('phone_number', _gen_phone),
    ('license_plate', _gen_license_plate),
    ('invoice_number', _gen_invoice),
    ('employee_id', _gen_employee_id),
    ('datetime', _gen_datetime),
    ('price_amount', _gen_price),
    ('discount_code', _gen_discount),
    ('zip_code', _gen_zip),
    ('serial_number', _gen_serial),
    ('api_key', _gen_apikey),
    ('version_number', _gen_version),
    ('temperature', _gen_temp),
    ('coordinate', _gen_coordinate),
    ('hex_color', _gen_hexcolor),
    ('flight_number', _gen_flight),
]


def build_group_a(rng: Random, n: int) -> List[Dict]:
    """n no-diacritic needles, 2 draws per category (20 categories x 2 = 40)."""
    out = []
    seen_payloads = set()
    reps_needed = -(-n // len(GROUP_A_GENERATORS))  # ceil
    for rep in range(reps_needed):
        for category, gen in GROUP_A_GENERATORS:
            if len(out) >= n:
                break
            for _ in range(20):  # regenerate on the (very unlikely) collision
                sentence, query, payload = gen(rng)
                if payload not in seen_payloads:
                    break
            seen_payloads.add(payload)
            assert not has_diacritic(payload), f"group A payload leaked a diacritic: {payload!r}"
            out.append({
                'needle_category': category,
                'needle': sentence,
                'query': query,
                'payload': payload,
            })
    return out[:n]


# ============================================================================
# Group B -- diacritic-bearing payloads (person names, place names, phrases)
# ============================================================================

SURNAMES = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ",
            "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý"]
MALE_MIDDLE = ["Văn", "Đức", "Quốc", "Hữu", "Xuân", "Công", "Thành", "Anh"]
FEMALE_MIDDLE = ["Thị", "Thị Mỹ", "Thị Thu", "Thị Ngọc", "Thị Kim", "Thị Bích"]
GIVEN_MALE = ["An", "Bình", "Cường", "Dũng", "Đạt", "Hải", "Hùng", "Khánh",
              "Long", "Minh", "Nam", "Phong", "Quang", "Sơn", "Thắng",
              "Thịnh", "Trung", "Tuấn", "Việt", "Vinh"]
GIVEN_FEMALE = ["Hà", "Hương", "Lan", "Linh", "Mai", "Ngọc", "Nhung",
                "Phương", "Quyên", "Thảo", "Trang", "Vân", "Yến", "Duyên",
                "Hoa", "Hiền", "Loan", "My", "Nga", "Xuân"]

NAME_ROLE_TEMPLATES = [
    ("Người phụ trách dự án", "Người phụ trách dự án là {title} {name}."),
    ("Trưởng ban tổ chức hội nghị", "Trưởng ban tổ chức hội nghị là {title} {name}."),
    ("Giám đốc kỹ thuật phụ trách hệ thống", "Giám đốc kỹ thuật phụ trách hệ thống là {title} {name}."),
    ("Chủ tịch hội đồng quản trị nhiệm kỳ này", "Chủ tịch hội đồng quản trị nhiệm kỳ này là {title} {name}."),
    ("Người ký hợp đồng đại diện bên bán", "Người ký hợp đồng đại diện bên bán là {title} {name}."),
    ("Trưởng nhóm nghiên cứu chính của đề tài", "Trưởng nhóm nghiên cứu chính của đề tài là {title} {name}."),
    ("Người phát ngôn chính thức của sự kiện", "Người phát ngôn chính thức của sự kiện là {title} {name}."),
    ("Kiến trúc sư trưởng của công trình", "Kiến trúc sư trưởng của công trình là {title} {name}."),
    ("Người phụ trách tiếp nhận hồ sơ", "Người phụ trách tiếp nhận hồ sơ là {title} {name}."),
    ("Huấn luyện viên trưởng đội tuyển", "Huấn luyện viên trưởng đội tuyển là {title} {name}."),
    ("Người chủ trì buổi lễ khánh thành", "Người chủ trì buổi lễ khánh thành là {title} {name}."),
    ("Trưởng phòng nhân sự phụ trách tuyển dụng", "Trưởng phòng nhân sự phụ trách tuyển dụng là {title} {name}."),
    ("Người đứng đầu đoàn thanh tra", "Người đứng đầu đoàn thanh tra là {title} {name}."),
    ("Chủ nhiệm khoa phụ trách chương trình", "Chủ nhiệm khoa phụ trách chương trình là {title} {name}."),
    ("Người được ủy quyền nhận hàng", "Người được ủy quyền nhận hàng là {title} {name}."),
    ("Trưởng đoàn công tác tại hiện trường", "Trưởng đoàn công tác tại hiện trường là {title} {name}."),
    ("Giám khảo chính của cuộc thi", "Giám khảo chính của cuộc thi là {title} {name}."),
    ("Người phụ trách truyền thông của chiến dịch", "Người phụ trách truyền thông của chiến dịch là {title} {name}."),
    ("Điều phối viên chính của dự án viện trợ", "Điều phối viên chính của dự án viện trợ là {title} {name}."),
    ("Người đại diện pháp lý của công ty", "Người đại diện pháp lý của công ty là {title} {name}."),
]

# A few otherwise-real Vietnamese place names have zero tone-marked
# syllables (e.g. "Nha Trang", "Sa Pa") -- excluded here since group B's
# entire point is rho > 0 under the actual w_tone metric (checked below,
# not just assumed from "looks Vietnamese").
_PLACE_POOL_CANDIDATES = ["Hội An", "Sa Pa", "Đà Lạt", "Ninh Bình", "Phú Quốc", "Huế",
                          "Nha Trang", "Vũng Tàu", "Hạ Long", "Mộc Châu", "Cát Bà",
                          "Côn Đảo", "Đồng Văn", "Tam Đảo", "Phong Nha", "Mai Châu",
                          "Bắc Hà", "Ba Vì", "Cần Thơ", "Đà Nẵng"]
PLACE_POOL = [p for p in _PLACE_POOL_CANDIDATES if has_diacritic(p)]

PLACE_TEMPLATES = [
    ("Địa điểm tổ chức lễ khai mạc", "Sự kiện khai mạc sẽ được tổ chức tại {place}."),
    ("Địa điểm tổ chức hội thảo chuyên đề năm nay", "Địa điểm tổ chức hội thảo chuyên đề năm nay là {place}."),
    ("Nơi diễn ra chuyến khảo sát thực địa tiếp theo", "Chuyến khảo sát thực địa tiếp theo sẽ diễn ra tại {place}."),
    ("Nơi đặt trụ sở chi nhánh mới", "Trụ sở chi nhánh mới được đặt tại {place}."),
    ("Nơi diễn ra lễ ký kết hợp tác song phương", "Lễ ký kết hợp tác song phương diễn ra tại {place}."),
    ("Điểm tập kết đoàn cứu trợ", "Điểm tập kết đoàn cứu trợ được chọn là {place}."),
    ("Nơi tổ chức trại huấn luyện mùa hè năm nay", "Trại huấn luyện mùa hè năm nay tổ chức tại {place}."),
    ("Nơi buổi triển lãm ảnh lưu động dừng chân", "Buổi triển lãm ảnh lưu động dừng chân tại {place}."),
    ("Địa điểm quay bộ phim tài liệu", "Địa điểm quay bộ phim tài liệu được chọn là {place}."),
    ("Nơi đặt trạm quan trắc môi trường mới", "Trạm quan trắc môi trường mới đặt tại {place}."),
]

PHRASE_TEMPLATES = [
    ("Phương châm hoạt động chính thức của đơn vị năm nay",
     "Phương châm hoạt động chính thức của đơn vị năm nay là “{phrase}”.", "Uống nước nhớ nguồn"),
    ("Khẩu hiệu chính của chiến dịch truyền thông",
     "Khẩu hiệu chính của chiến dịch truyền thông là “{phrase}”.", "Ăn quả nhớ kẻ trồng cây"),
    ("Câu châm ngôn được treo trang trọng tại hội trường",
     "Câu châm ngôn được treo trang trọng tại hội trường là “{phrase}”.", "Lá lành đùm lá rách"),
    ("Thông điệp chủ đạo của chương trình năm nay",
     "Thông điệp chủ đạo của chương trình năm nay là “{phrase}”.", "Có công mài sắt có ngày nên kim"),
    ("Câu tục ngữ được nhắc đến trong bài phát biểu khai mạc",
     "Câu tục ngữ được nhắc đến trong bài phát biểu khai mạc là “{phrase}”.", "Đoàn kết là sức mạnh"),
    ("Khẩu hiệu treo tại cổng chào của lễ hội",
     "Khẩu hiệu treo tại cổng chào của lễ hội là “{phrase}”.", "Học đi đôi với hành"),
    ("Câu nói được chọn làm phương châm của lớp học",
     "Câu nói được chọn làm phương châm của lớp học là “{phrase}”.", "Tương thân tương ái"),
    ("Thông điệp in trên banner chính của sự kiện",
     "Thông điệp in trên banner chính của sự kiện là “{phrase}”.", "Kính trên nhường dưới"),
    ("Câu danh ngôn được trích dẫn mở đầu tài liệu",
     "Câu danh ngôn được trích dẫn mở đầu tài liệu là “{phrase}”.", "Một cây làm chẳng nên non"),
    ("Khẩu hiệu vận động được sử dụng trong chiến dịch",
     "Khẩu hiệu vận động được sử dụng trong chiến dịch là “{phrase}”.", "Chậm mà chắc"),
]


def build_group_b(rng: Random, n: int) -> List[Dict]:
    """n diacritic-bearing needles: 20 person-name + 10 place + 10 phrase."""
    n_names = n - len(PLACE_TEMPLATES) - len(PHRASE_TEMPLATES)
    assert n_names == len(NAME_ROLE_TEMPLATES), "adjust pools if n != 40"

    out = []
    names_used = set()
    for role_desc, template in NAME_ROLE_TEMPLATES:
        while True:
            is_male = rng.random() < 0.5
            surname = rng.choice(SURNAMES)
            middle = rng.choice(MALE_MIDDLE if is_male else FEMALE_MIDDLE)
            given = rng.choice(GIVEN_MALE if is_male else GIVEN_FEMALE)
            name = f"{surname} {middle} {given}"
            # Some real Vietnamese names have no tone-marked syllable at all
            # (e.g. "Phan Anh Minh" -- Ê/Ô/Ơ/Ư/Ă/Â/Đ are base letters, not
            # tone marks, so rho() can still be 0). Group B's entire point is
            # rho > 0, so require it here rather than assume it from "looks
            # Vietnamese".
            if name not in names_used and has_diacritic(name):
                names_used.add(name)
                break
        title = "ông" if is_male else "bà"
        sentence = template.format(title=title, name=name)
        query = f"{role_desc} được nhắc đến trong đoạn văn là ai?"
        out.append({
            'needle_category': 'person_name',
            'needle': sentence,
            'query': query,
            'payload': name,
        })

    places = rng.sample(PLACE_POOL, len(PLACE_TEMPLATES))
    for (desc, template), place in zip(PLACE_TEMPLATES, places):
        sentence = template.format(place=place)
        query = f"{desc} được nhắc đến trong đoạn văn là ở đâu?"
        out.append({
            'needle_category': 'place_name',
            'needle': sentence,
            'query': query,
            'payload': place,
        })

    for desc, template, phrase in PHRASE_TEMPLATES:
        sentence = template.format(phrase=phrase)
        query = f"{desc} được nhắc đến trong đoạn văn là gì?"
        out.append({
            'needle_category': 'phrase',
            'needle': sentence,
            'query': query,
            'payload': phrase,
        })

    for item in out:
        assert has_diacritic(item['payload']), f"group B payload has no diacritic: {item['payload']!r}"
    return out[:n]


# ============================================================================
# Group C -- mixed: one no-diacritic fact + one diacritic-bearing fact
# ============================================================================

DOC_TYPES = [
    ("Hợp đồng", "HĐ"), ("Quyết định", "QĐ"), ("Biên bản", "BB"),
    ("Công văn", "CV"), ("Thông báo", "TB"), ("Giấy chứng nhận", "GCN"),
    ("Phiếu thu", "PT"), ("Đơn đặt hàng", "DH"), ("Giấy ủy quyền", "UQ"),
    ("Tờ trình", "TT"),
]


def _gen_c_code(rng: Random, prefix: str) -> str:
    return f"{prefix}-{rng.randint(2026, 2027)}-{_digits(rng, 3)}"


def build_group_c(rng: Random, n: int) -> List[Dict]:
    """n mixed needles: half pair a code with a person name, half with a
    place -- reusing DOC_TYPES x 4 variants each side (10 x 2 x 2 = 40)."""
    half = n // 2
    out = []
    names_used = set()

    def fresh_name():
        while True:
            is_male = rng.random() < 0.5
            surname = rng.choice(SURNAMES)
            middle = rng.choice(MALE_MIDDLE if is_male else FEMALE_MIDDLE)
            given = rng.choice(GIVEN_MALE if is_male else GIVEN_FEMALE)
            name = f"{surname} {middle} {given}"
            if name not in names_used and has_diacritic(name):
                names_used.add(name)
                return name, ("ông" if is_male else "bà")

    variants_per_type = -(-half // len(DOC_TYPES))
    made = 0
    for _ in range(variants_per_type):
        for doc_type, prefix in DOC_TYPES:
            if made >= half:
                break
            code = _gen_c_code(rng, prefix)
            name, title = fresh_name()
            day = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2026, 2027)}"
            sentence = f"{doc_type} số {code} do {title} {name} ký ngày {day}."
            query = f"{doc_type} được nhắc đến trong đoạn văn mang số hiệu gì và do ai ký?"
            payload = f"{code}, {title} {name}"
            out.append({
                'needle_category': 'mixed_code_name',
                'needle': sentence,
                'query': query,
                'payload': payload,
                'needle_payload_parts': {'no_diacritic_part': code, 'diacritic_part': f"{title} {name}"},
            })
            made += 1

    places_cycle = PLACE_POOL * (-(-(n - half) // len(PLACE_POOL)))
    rng.shuffle(places_cycle)
    made = 0
    for _ in range(-(-(n - half) // len(DOC_TYPES))):
        for doc_type, prefix in DOC_TYPES:
            if made >= (n - half):
                break
            code = _gen_c_code(rng, prefix)
            place = places_cycle[made]
            day = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2026, 2027)}"
            sentence = f"{doc_type} số {code} liên quan khu vực {place} được ban hành ngày {day}."
            query = f"{doc_type} được nhắc đến trong đoạn văn mang số hiệu gì và liên quan khu vực nào?"
            payload = f"{code}, {place}"
            out.append({
                'needle_category': 'mixed_code_place',
                'needle': sentence,
                'query': query,
                'payload': payload,
                'needle_payload_parts': {'no_diacritic_part': code, 'diacritic_part': place},
            })
            made += 1

    for item in out:
        assert has_diacritic(item['payload']), f"group C payload has no diacritic: {item['payload']!r}"
    return out[:n]


# ============================================================================
# Shared haystack construction
# ============================================================================

def build_base_haystack_text(paragraphs: List[Dict], rng: Random, target_len: int) -> str:
    """A haystack of ~target_len chars, built from a fresh shuffle of the
    Wikipedia paragraph pool, truncated (at a word boundary) to hit the
    target closely regardless of which paragraph sizes got drawn -- this is
    what keeps context length within <5% across all 120 needle samples
    (Việc 1, variable #2), independent of needle length (added on top)."""
    pool = paragraphs[:]
    rng.shuffle(pool)
    parts: List[str] = []
    total = 0
    for p in pool:
        text = p['text']
        remaining = target_len - total
        if remaining <= 0:
            break
        if len(text) <= remaining:
            parts.append(text)
            total += len(text) + 2  # '\n\n' separator
        else:
            cut = text[:remaining]
            sp = cut.rfind(' ')
            if sp > remaining * 0.5:
                cut = cut[:sp]
            cut = cut.rstrip().rstrip('.,;:') + '.'
            parts.append(cut)
            total += len(cut) + 2
            break
    return '\n\n'.join(parts)


def insert_needle(base_text: str, needle_sentence: str, position: str) -> Tuple[str, int, int]:
    parts = base_text.split('\n\n')
    if position == 'beginning':
        idx = 0
    elif position == 'end':
        idx = len(parts)
    else:
        idx = len(parts) // 2
    new_parts = parts[:idx] + [needle_sentence] + parts[idx:]
    full = '\n\n'.join(new_parts)
    offset = full.index(needle_sentence)
    return full, offset, len(new_parts)


def build_needle_v2(wiki_paragraphs: List[Dict], rng: Random) -> List[Dict]:
    group_a = build_group_a(rng, NUM_HAYSTACKS)
    group_b = build_group_b(rng, NUM_HAYSTACKS)
    group_c = build_group_c(rng, NUM_HAYSTACKS)

    # Even 3-way split of insert positions per haystack triple (14/13/13),
    # order shuffled so position doesn't correlate with haystack_id/content.
    positions = (['beginning'] * 14 + ['middle'] * 13 + ['end'] * 13)
    rng.shuffle(positions)

    samples = []
    for h in range(NUM_HAYSTACKS):
        haystack_id = f"hay_{h:04d}"
        base_text = build_base_haystack_text(wiki_paragraphs, rng, TARGET_CONTEXT_LEN)
        base_len = len(base_text)
        position = positions[h]

        for group_letter, group_data, needle_item in (
            ('A', group_a, group_a[h]),
            ('B', group_b, group_b[h]),
            ('C', group_c, group_c[h]),
        ):
            full_context, offset, num_paragraphs = insert_needle(base_text, needle_item['needle'], position)
            payload = needle_item['payload']
            sample = {
                'sample_id': f"needle_v2_{h:04d}_{group_letter}",
                'task': 'needle_in_haystack',
                'needle_group': group_letter,
                'needle_category': needle_item['needle_category'],
                'needle_has_diacritic': has_diacritic(payload),
                'needle_payload': payload,
                'needle_char_len': len(payload),
                'insert_position': position,
                'insert_char_offset': offset,
                'haystack_id': haystack_id,
                'haystack_base_char_len': base_len,
                'context': full_context,
                'query': needle_item['query'],
                'reference_answer': payload,
                'needle': needle_item['needle'],
                'char_length': len(full_context),
                'num_paragraphs': num_paragraphs,
                'synthetic_pii': True,
            }
            if 'needle_payload_parts' in needle_item:
                sample['needle_payload_parts'] = needle_item['needle_payload_parts']
            samples.append(sample)

    return samples


# ============================================================================
# long_document_qa additions: multi_hop + referential
# ============================================================================

REFERENTIAL_MARKERS = [
    "Điều này", "điều này", "Điều đó", "điều đó", "Nhờ đó", "nhờ đó",
    "Do đó", "do đó", "Vì vậy", "vì vậy", "Nhờ vậy", "nhờ vậy",
    "Trong khi đó", "trong khi đó", "Tuy nhiên", "tuy nhiên",
    "Ông ", "Bà ", "ông ", "bà ",
]


def _group_paragraphs_by_article(paragraphs: List[Dict]) -> Dict[str, List[Dict]]:
    by_topic: Dict[str, List[Dict]] = {}
    for p in paragraphs:
        by_topic.setdefault(p['topic_id'], []).append(p)
    for arts in by_topic.values():
        arts.sort(key=lambda x: x['paragraph_index'])
    return by_topic


def _snippet(text: str, n_words: int = 8) -> str:
    words = text.split()
    return ' '.join(words[:n_words]) + ('…' if len(words) > n_words else '')


# Some articles (e.g. Ha_Noi, 69 paragraphs) run past 100k chars in full --
# using the whole thing would make multi_hop/referential samples wildly
# longer than every other task (including needle_in_haystack's controlled
# ~30k) for no benefit to the evidence-retention question being tested. Cap
# how many paragraphs go into context, but always keep the evidence
# paragraph(s) plus a rng-sampled spread of distractor paragraphs from the
# same article, so compression still has to traverse real distance between
# the evidence pieces.
LONG_DOC_QA_PARAGRAPH_CAP = 20


def _bounded_context(arts: List[Dict], must_include: List[int], rng: Random) -> Tuple[str, List[int]]:
    """Return (context, local_evidence_indices) for a paragraph subset of at
    most LONG_DOC_QA_PARAGRAPH_CAP paragraphs, in original document order,
    guaranteed to contain every index in must_include."""
    if len(arts) <= LONG_DOC_QA_PARAGRAPH_CAP:
        chosen_indices = list(range(len(arts)))
    else:
        must = set(must_include)
        others = [i for i in range(len(arts)) if i not in must]
        take = max(0, LONG_DOC_QA_PARAGRAPH_CAP - len(must))
        chosen_indices = sorted(must | set(rng.sample(others, min(take, len(others)))))
    context = '\n\n'.join(arts[i]['text'] for i in chosen_indices)
    local = [chosen_indices.index(i) for i in must_include]
    return context, local


def build_multi_hop(wiki_paragraphs: List[Dict], rng: Random, target_n: int) -> List[Dict]:
    by_article = _group_paragraphs_by_article(wiki_paragraphs)
    qualifying = {t: arts for t, arts in by_article.items() if len(arts) >= 6}

    samples = []
    idx = 0
    article_ids = sorted(qualifying.keys())
    while len(samples) < target_n:
        progressed = False
        for topic_id in article_ids:
            if len(samples) >= target_n:
                break
            arts = qualifying[topic_id]
            n = len(arts)
            gap = max(2, n // 5)
            i = rng.randint(0, n - 1 - gap)
            j = rng.randint(i + gap, n - 1)
            title = arts[0]['title']
            context, (local_i, local_j) = _bounded_context(arts, [i, j], rng)
            snippet_i = _snippet(arts[i]['text'])
            snippet_j = _snippet(arts[j]['text'])
            query = (f"Kết hợp thông tin ở đoạn bắt đầu bằng “{snippet_i}” và đoạn bắt đầu bằng "
                     f"“{snippet_j}” trong bài viết về {title}, hai đoạn này cùng cho biết điều gì?")
            reference_answer = arts[i]['text'] + ' ' + arts[j]['text']
            samples.append({
                'sample_id': f"doc_qa_multihop_{idx:04d}",
                'task': 'long_document_qa',
                'qa_subset': 'multi_hop',
                'title': title,
                'context': context,
                'query': query,
                'reference_answer': reference_answer,
                'evidence_paragraph_indices': [local_i, local_j],
                'domain': 'general',
                'source': 'wikipedia',
                'char_length': len(context),
            })
            idx += 1
            progressed = True
        if not progressed:
            break
    return samples[:target_n]


def build_referential(wiki_paragraphs: List[Dict], rng: Random, target_n: int) -> List[Dict]:
    by_article = _group_paragraphs_by_article(wiki_paragraphs)
    qualifying = {t: arts for t, arts in by_article.items() if len(arts) >= 6}

    candidates = []
    for topic_id, arts in qualifying.items():
        for r, para in enumerate(arts[1:], start=1):
            if any(m in para['text'] for m in REFERENTIAL_MARKERS):
                candidates.append((topic_id, arts, r))
    rng.shuffle(candidates)

    samples = []
    idx = 0
    seen_per_article: Dict[str, int] = {}
    for topic_id, arts, r in candidates:
        if len(samples) >= target_n:
            break
        # Spread across articles instead of exhausting one article's markers first.
        if seen_per_article.get(topic_id, 0) >= 3:
            continue
        seen_per_article[topic_id] = seen_per_article.get(topic_id, 0) + 1

        title = arts[0]['title']
        context, (local_0, local_r) = _bounded_context(arts, [0, r], rng)
        query = (f"Trong bài viết về {title}, đoạn văn sau phần giới thiệu có nhắc lại chủ thể đã nêu ở đoạn mở "
                 f"đầu bằng đại từ hoặc cách chỉ định gián tiếp (thay vì lặp lại tên) -- chủ thể đó là gì/ai, "
                 f"và nó được giới thiệu lần đầu ở đoạn nào?")
        reference_answer = arts[0]['text'] + ' ' + arts[r]['text']
        samples.append({
            'sample_id': f"doc_qa_referential_{idx:04d}",
            'task': 'long_document_qa',
            'qa_subset': 'referential',
            'title': title,
            'context': context,
            'query': query,
            'reference_answer': reference_answer,
            'evidence_paragraph_indices': [local_0, local_r],
            'domain': 'general',
            'source': 'wikipedia',
            'char_length': len(context),
        })
        idx += 1
    return samples


# ============================================================================
# Main
# ============================================================================

def main():
    rng = Random(SEED)

    print('=' * 60)
    print('VCC-Bench v2 Builder')
    print('=' * 60)

    v1_path = os.path.join(DATA_DIR, 'vcc_bench_v1.json')
    with open(v1_path, 'r', encoding='utf-8') as f:
        v1 = json.load(f)
    v1_samples = v1['samples']
    print(f"\n[1/5] Loaded {len(v1_samples)} v1 samples from {v1_path} (frozen, unchanged)")

    doc_qa_v1 = [s for s in v1_samples if s['task'] == 'long_document_qa']
    for s in doc_qa_v1:
        s.setdefault('qa_subset', 'standard')
    other_v1 = [s for s in v1_samples
                if s['task'] in ('multi_turn_conversation', 'agent_tool_calling', 'cross_lingual')]
    print(f"  long_document_qa carried over: {len(doc_qa_v1)}")
    print(f"  other tasks carried over unchanged: {len(other_v1)} "
          f"({', '.join(sorted(set(s['task'] for s in other_v1)))})")

    print('\n[2/5] Loading Wikipedia paragraphs...')
    wiki_paragraphs = load_wikipedia_data()
    print(f"  {len(wiki_paragraphs)} paragraphs")

    print('\n[3/5] Building needle-in-haystack v2 (120 samples, groups A/B/C)...')
    needle_samples = build_needle_v2(wiki_paragraphs, rng)
    print(f"  {len(needle_samples)} samples "
          f"(A={sum(1 for s in needle_samples if s['needle_group']=='A')}, "
          f"B={sum(1 for s in needle_samples if s['needle_group']=='B')}, "
          f"C={sum(1 for s in needle_samples if s['needle_group']=='C')})")

    lengths = [s['char_length'] for s in needle_samples]
    spread = (max(lengths) - min(lengths)) / (sum(lengths) / len(lengths))
    print(f"  char_length min={min(lengths)} max={max(lengths)} mean={sum(lengths)//len(lengths)} "
          f"spread={spread:.2%} (must be <5%)")
    assert spread < 0.05, f"needle context length spread {spread:.2%} exceeds 5%"

    print('\n[4/5] Building long_document_qa additions (multi_hop, referential)...')
    multi_hop = build_multi_hop(wiki_paragraphs, rng, target_n=30)
    referential = build_referential(wiki_paragraphs, rng, target_n=30)
    print(f"  multi_hop: {len(multi_hop)}, referential: {len(referential)}")

    all_doc_qa = doc_qa_v1 + multi_hop + referential
    all_samples = all_doc_qa + other_v1 + needle_samples

    print('\n[5/5] Validating combined dataset...')
    stats = validate_dataset(all_samples)
    print(f"  Total samples: {stats['total_samples']}")
    for task, count in sorted(stats['task_distribution'].items()):
        print(f"    {task}: {count}")
    if stats['issues']:
        print(f"  Issues ({len(stats['issues'])}):")
        for issue in stats['issues'][:10]:
            print(f"    - {issue}")
    else:
        print('  No quality issues found.')

    haystack_ids = sorted(set(s['haystack_id'] for s in needle_samples))
    dataset = {
        'metadata': {
            'name': 'VCC-Bench v2.0',
            'version': '2.0.0',
            'date': time.strftime('%Y-%m-%d'),
            'language': 'vi',
            'description': 'Vietnamese Context Compression Benchmark v2 -- controlled needle-in-haystack '
                            '(diacritic A/B/C groups) + multi-hop/referential long_document_qa subsets. '
                            'See docs/build_VCC-Bench v2.md.',
            'tasks': sorted(list(stats['task_distribution'].keys())),
            'total_samples': len(all_samples),
            'license': 'CC-BY-SA 4.0 (Wikipedia) + Public Domain (Legal) + MIT (synthetic)',
            'needle_seed': SEED,
            'needle_haystack_ids': haystack_ids,
            'supersedes': 'vcc_bench_v1.json (needle_in_haystack, long_document_qa only -- '
                          'multi_turn_conversation/agent_tool_calling/cross_lingual carried over unchanged)',
        },
        'statistics': stats,
        'samples': all_samples,
    }

    out_path = os.path.join(DATA_DIR, 'vcc_bench_v2.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")


if __name__ == '__main__':
    main()
