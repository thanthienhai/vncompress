#!/usr/bin/env python3
"""
Fetch real Vietnamese text data for VCC-Bench from Wikipedia and legal sources.
"""
import json
import os
import sys
import time
import urllib.request
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

USER_AGENT = 'VCCBench/1.0 (research project; contact@example.com)'
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vcc_bench_data')
os.makedirs(DATA_DIR, exist_ok=True)

WIKI_TOPICS = [
    'Lich_su_Viet_Nam', 'Kinh_te_Viet_Nam', 'Van_hoa_Viet_Nam',
    'Giao_duc_Viet_Nam', 'Luat_phap_Viet_Nam', 'Y_te_Viet_Nam',
    'Cong_nghe_thong_tin', 'Tri_tue_nhan_tao', 'Bien_doi_khi_hau',
    'Nang_luong_tai_tao', 'Vat_ly_hoc', 'Sinh_hoc',
    'Van_hoc_Viet_Nam', 'Am_nhac_Viet_Nam',
    'Ha_Noi', 'Thanh_pho_Ho_Chi_Minh', 'Song_Hong',
    'Bien_Dong', 'Ton_giao_tai_Viet_Nam', 'Am_thuc_Viet_Nam',
    'Hien_phap_Viet_Nam', 'Quoc_hoi_Viet_Nam', 'Chinh_phu_Viet_Nam',
    'Nong_nghiep_Viet_Nam', 'Giao_thong_Viet_Nam',
]

VIETNAMESE_DISPLAY_NAMES = {
    'Lich_su_Viet_Nam': 'Lịch sử Việt Nam',
    'Kinh_te_Viet_Nam': 'Kinh tế Việt Nam',
    'Van_hoa_Viet_Nam': 'Văn hóa Việt Nam',
    'Giao_duc_Viet_Nam': 'Giáo dục Việt Nam',
    'Luat_phap_Viet_Nam': 'Luật pháp Việt Nam',
    'Y_te_Viet_Nam': 'Y tế Việt Nam',
    'Cong_nghe_thong_tin': 'Công nghệ thông tin',
    'Tri_tue_nhan_tao': 'Trí tuệ nhân tạo',
    'Bien_doi_khi_hau': 'Biến đổi khí hậu',
    'Nang_luong_tai_tao': 'Năng lượng tái tạo',
    'Vat_ly_hoc': 'Vật lý học',
    'Sinh_hoc': 'Sinh học',
    'Van_hoc_Viet_Nam': 'Văn học Việt Nam',
    'Am_nhac_Viet_Nam': 'Âm nhạc Việt Nam',
    'Ha_Noi': 'Hà Nội',
    'Thanh_pho_Ho_Chi_Minh': 'Thành phố Hồ Chí Minh',
    'Song_Hong': 'Sông Hồng',
    'Bien_Dong': 'Biển Đông',
    'Ton_giao_tai_Viet_Nam': 'Tôn giáo tại Việt Nam',
    'Am_thuc_Viet_Nam': 'Ẩm thực Việt Nam',
    'Hien_phap_Viet_Nam': 'Hiến pháp Việt Nam',
    'Quoc_hoi_Viet_Nam': 'Quốc hội Việt Nam',
    'Chinh_phu_Viet_Nam': 'Chính phủ Việt Nam',
    'Nong_nghiep_Viet_Nam': 'Nông nghiệp Việt Nam',
    'Giao_thong_Viet_Nam': 'Giao thông Việt Nam',
}


def fetch_wikipedia_article(ascii_title: str, display_title: str) -> dict:
    """Fetch full text of a Vietnamese Wikipedia article."""
    encoded = urllib.parse.quote(display_title.replace(' ', '_'))
    url = (
        f'https://vi.wikipedia.org/w/api.php'
        f'?action=query&format=json&titles={encoded}'
        f'&prop=extracts&explaintext=1&exsectionformat=plain'
    )
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                pages = data.get('query', {}).get('pages', {})
                for pid, page in pages.items():
                    if pid == '-1':
                        return None
                    text = page.get('extract', '')
                    if len(text) < 500:
                        return None
                    return {
                        'topic_id': ascii_title,
                        'title': page.get('title', display_title),
                        'text': text,
                        'char_length': len(text),
                        'source': 'wikipedia',
                        'url': f'https://vi.wikipedia.org/wiki/{encoded}',
                    }
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                print(f'  FAIL [{ascii_title}]: {e}', file=sys.stderr)
    return None


def segment_paragraphs(text: str, min_chars: int = 200) -> list:
    """Split text into meaningful paragraphs."""
    paragraphs = []
    current = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            if current:
                para = ' '.join(current)
                if len(para) >= min_chars:
                    paragraphs.append(para)
                current = []
        else:
            current.append(line)
    if current:
        para = ' '.join(current)
        if len(para) >= min_chars:
            paragraphs.append(para)
    return paragraphs


def main():
    print('=' * 60)
    print('VCC-Bench Data Fetcher')
    print('=' * 60)
    print(f'Fetching {len(WIKI_TOPICS)} Wikipedia articles...\n')

    articles = {}
    for i, ascii_title in enumerate(WIKI_TOPICS):
        display = VIETNAMESE_DISPLAY_NAMES.get(ascii_title, ascii_title)
        print(f'[{i+1}/{len(WIKI_TOPICS)}] {display}...', end=' ', flush=True)
        result = fetch_wikipedia_article(ascii_title, display)
        if result:
            articles[ascii_title] = result
            print(f'OK ({result["char_length"]} chars)')
        else:
            print('SKIP (not found or too short)')
        time.sleep(0.5)

    print(f'\nFetched {len(articles)} articles successfully.')

    all_paragraphs = []
    for aid, article in articles.items():
        paragraphs = segment_paragraphs(article['text'])
        for pi, para in enumerate(paragraphs):
            all_paragraphs.append({
                'source': 'wikipedia',
                'topic_id': aid,
                'title': article['title'],
                'paragraph_index': pi,
                'text': para,
                'char_length': len(para),
            })

    print(f'Extracted {len(all_paragraphs)} total paragraphs.')

    lengths = [p['char_length'] for p in all_paragraphs]
    if lengths:
        print(f'  Paragraph stats: min={min(lengths)}, max={max(lengths)}, '
              f'avg={sum(lengths)//len(lengths)}, median={sorted(lengths)[len(lengths)//2]}')

    output = {
        'metadata': {
            'version': '1.0.0',
            'date': time.strftime('%Y-%m-%d'),
            'source': 'vi.wikipedia.org (Wikimedia API)',
            'license': 'CC-BY-SA 4.0',
            'num_articles': len(articles),
            'num_paragraphs': len(all_paragraphs),
            'article_topics': list(articles.keys()),
        },
        'articles': articles,
        'paragraphs': all_paragraphs,
    }

    out_path = os.path.join(DATA_DIR, 'wikipedia_vi_raw.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\nSaved to: {out_path}')
    print(f'File size: {os.path.getsize(out_path) / 1024:.1f} KB')


if __name__ == '__main__':
    main()
