"""
Source-derived, condition-aware answers for recognized tariff operations.
A tariff must match the requested operation, geography, channel, currency and limit.
The full fee string (including +percentage, min/max) is taken from the source row, never calculated or embedded in application code.
"""
from __future__ import annotations

import re
from typing import Any
from src.extractive import response
from src.query_policy import norm

Hit = dict[str, Any]
FEE = 'fee_schedule_20260901'
NOTICE = 'notice_20251118'
GLOSS = 'glossary_undated'


def section(hits: list[Hit], doc: str, sid: str) -> Hit | None:
    return next((h for h in hits if h.get('metadata',{}).get('doc_id') == doc
                 and h.get('metadata',{}).get('section_id') == sid), None)


def row(hit: Hit | None, *words: str) -> tuple[str, list[str]] | None:
    if not hit: return None
    for line in hit['content'].splitlines():
        if not line.startswith('|') or '---' in line: continue
        cells = [x.strip() for x in line.strip('|').split('|')]
        if all(norm(w) in norm(' '.join(cells[:-1])) for w in words):
            return line, cells
    return None


def bullet(hit: Hit | None, *words: str) -> str | None:
    if not hit: return None
    for line in hit['content'].splitlines():
        if line.startswith('- ') and all(norm(w) in norm(line) for w in words):
            return line
    return None


def last_fee(line: str) -> str:
    if '|' in line: return line.strip().strip('|').split('|')[-1].strip()
    return line.split(':',1)[-1].strip().rstrip(';').rstrip('.')


def supported(text: str, items: list[tuple[Hit,str]], mode: str = 'condition_checked') -> dict:
    result=response(text, items)
    result['mode']=mode
    return result


def not_documented(text: str, items: list[tuple[Hit,str]] | None = None) -> dict:
    result = supported(text, items or [], mode='document_limit_guard') if items else {
        'answer': text, 'sources': [], 'evidence': [], 'mode': 'document_limit_guard'
    }
    result['status']='insufficient'
    return result


def _amount(question: str) -> int | None:
    m = re.search(r'\b(\d{1,3}(?:[\s\u00a0]\d{3})+|\d+)\s*(?:Ft|forint)', question, re.I)
    return int(re.sub(r'\s','',m.group(1))) if m else None


def _fee_from_bullet(hit: Hit | None, desc: str, keywords: tuple[str, ...]) -> tuple[str,tuple[Hit,str]] | None:
    line=bullet(hit,*keywords)
    return (f'{desc}: {last_fee(line)}', (hit,line)) if line and hit else None


def _sms(question: str, hits: list[Hit]) -> dict | None:
    q=norm(question)
    if not ('sms' in q or 'ertesito' in q): return None
    is_package=any(w in q for w in ('csomag','havi dij','eves dij'))
    has_message=any(w in q for w in ('ertesit', 'uzenet', 'telekom', 'darabonkent', 'sms is'))
    result_parts=[]
    if is_package:
        h=section(hits,FEE,'SRC-P04-B03')
        monthly=bullet(h,'Havi díj:');yearly=bullet(h,'Teljes éves díj:')
        if h and monthly and yearly:
            result_parts.append((f'Az SMS-szolgáltatás csomagdíja havonta {last_fee(monthly)}, évente {last_fee(yearly)}.',[(h,monthly),(h,yearly)]))
    if has_message:
        h=section(hits,FEE,'SRC-P04-B02')
        operator='Magyar Telekom' if 'telekom' in q else ('Yettel' if 'yettel' in q else ('Vodafone-One' if 'vodafone' in q else None))
        if operator:
            line=bullet(h,operator)
            if line and h:
                result_parts.append((f'Az értesítő/egyéb SMS díja a {operator} hálózatába: {last_fee(line)} üzenetenként.',[(h,line)]))
        elif not is_package:
            return None
    if not result_parts: return None
    return supported(' '.join(p for p,_ in result_parts), [item for _, pairs in result_parts for item in pairs])


def _atm(question: str,hits: list[Hit]) -> dict | None:
    q=norm(question)
    if not ('keszpenzfelvet' in q or ('atm' in q and any(x in q for x in ('felv','kivet','veszek fel','venni','felven')))):
        return None
    foreign=any(t in q for t in ('kulfold','nemetorszag','ausztria','egt','franciaorszag','olaszorszag','spanyolorszag'))
    other=any(t in q for t in ('idegen','masik bank','mas bank','nem otp','harmadik')) or (('otp' in q or 'diakhitel' in q) and any(t in q for t in ('mbh','erste','raiffeisen','unicredit','revolut','k&h')) and 'atm' in q)
    own=any(t in q for t in ('sajat','otp-s','otp atm')) and not other
    if 'hitelkartya' in q:
        h=section(hits,FEE,'SRC-P07-B01')
        if h and 'nem áll rendelkezésre' in h['content']:
            line=next((x for x in h['content'].splitlines() if 'nem áll rendelkezésre' in x), None)
            if line:return supported('Az OTP Diákhitel Számla díjjegyzékében a hitelkártya-szolgáltatás nem áll rendelkezésre.',[(h,line)])
    if not foreign:
        h=section(hits,FEE,'SRC-P07-B04')
        if not own and not other:
            return not_documented('A belföldi ATM-díjhoz meg kell adni, hogy saját vagy idegen ATM-ről van-e szó.')
        chosen=row(h, 'Betéti bankkártyával', 'saját ATM-ből' if own else 'idegen ATM-ből')
        if not chosen:return None
        quote, cells=chosen; fee=cells[-1]
        if own:
            m=re.search(r'(\d{1,3}(?:[\s\u00a0]\d{3})+)\s*Ft\s+tranzakcióösszegig',cells[1])
            cap=int(re.sub(r'\s','',m.group(1))) if m else None
            amount=_amount(question)
            if cap and amount is not None and amount > cap:
                return not_documented(f'A forrás a saját ATM-es készpénzfelvétel díját csak {cap:,} Ft tranzakcióösszegig adja meg; a kérdezett {amount:,} Ft-ra nem közöl alkalmazható díjat.'.replace(',',' '),[(h,quote)])
            return supported(f'Belföldön, betéti kártyával, saját ATM-ből {cells[1]} a díj {fee}.',[(h,quote)])
        return supported(f'Belföldön idegen ATM-ből betéti kártyával végzett készpénzfelvétel díja {fee}.',[(h,quote)])
    h=section(hits,FEE,'SRC-P08-B01')
    if not h:return None
    in_eea=any(t in q for t in ('nemetorszag','ausztria','egt','franciaorszag','olaszorszag','spanyolorszag'))
    euro=bool(re.search(r'\b(?:euro|eur|euroban|eurot)\b',q))
    outside_eea=any(t in q for t in ('egt-n kivul','egt-n kivul', 'egt allamokon kivul','usa','egyesult allamok','anglia','torokorszag'))
    non_euro=any(t in q for t in ('nem euro','nem euroban','forintban','dollar','usd','gbp'))
    if other:
        quote=bullet(h,'Idegen ATM-ből')
        if quote and (in_eea and euro and not non_euro):
            return not_documented('A díjjegyzék az EGT-államokban euróban, idegen ATM-nél végzett készpénzfelvétel díját nem adja meg; a feltüntetett díj más pénznemű EGT- vagy EGT-n kívüli tranzakciókra vonatkozik.',[(h,quote)])
        if not (outside_eea or (in_eea and non_euro)):
            return not_documented('Az idegen külföldi ATM díjához a helyszín és a pénznem feltételeit is igazolni kell; a rendelkezésre álló díjsor nem általános.', [(h,quote)] if quote else None)
        return supported(f'Az idegen külföldi ATM díja a megadott földrajzi és pénznemfeltételek mellett {last_fee(quote)}.',[(h,quote)]) if quote else None
    if own:
        quote=bullet(h,'saját ATM-ből')
        if quote:return supported(f'A számlavezetői csoport saját külföldi ATM-jének feltüntetett díja {last_fee(quote)}.',[(h,quote)])
    return not_documented('Külföldi készpénzfelvételhez meg kell adni a saját/idegen ATM-et, a helyszínt és a pénznemet.')


def _transfer(question: str,hits:list[Hit]) -> dict | None:
    q=norm(question)
    if 'sepa' in q:
        h=section(hits,FEE,'SRC-P06-B01')
        if not h:return None
        paper=any(t in q for t in ('fiok','papir','szemelyesen'))
        electronic=any(t in q for t in ('mobilbank','elektronikus','online','internetbank'))
        telephone=('telefon' in q)
        # A request mentioning multiple channels is comparative; the single- channel resolver must not claim that one row answers both.
        if sum((paper,electronic,telephone)) != 1:
            return None
        channel_key = 'Papíralapon' if paper else ('Telefonon' if telephone else 'Elektronikusan')
        outside=any(t in q for t in ('kulfold', 'nemetorszag', 'ausztria'))
        domestic='belfold' in q
        if not (outside or domestic):
            # Neither direction was stated; only return a tariff when *both* documented directions give the same fee for this channel.
            choices=[row(h,'Külföldre',channel_key), row(h,'Belföldre',channel_key)]
            if not all(choices) or choices[0][1][-1]!=choices[1][1][-1]:return None
            chosen=choices[0]
        else:
            chosen=row(h,'Külföldre' if outside else 'Belföldre',channel_key)
        if chosen:
            quote,cells=chosen
            channel='papíralapon, bankfiókban' if paper else ('telefonon' if telephone else 'elektronikusan')
            return supported(f'A SEPA euróátutalás {channel} megadott díja {cells[-1]}.',[(h,quote)])
    if 'utal' in q or 'atutal' in q:
        regular='rendszeres' in q
        paper=any(t in q for t in ('fiok','papir','szemelyesen'))
        domestic=any(t in q for t in ('belfold','forint'))
        if regular:
            h=section(hits,FEE,'SRC-P05-B02')
            quote=next((x for x in h['content'].splitlines() if '0 Ft' in x),None) if h else None
            if quote:return supported('A díjjegyzék szerint a felsorolt rendszeres forintátutalások díja 0 Ft.',[(h,quote)])
        if paper and domestic:
            h=section(hits,FEE,'SRC-P05-B01')
            outside=any(x in q for x in ('masik bank','bankon kivul','idegen bank'))
            chosen=row(h,'Számlavezetőn kívül' if outside else 'Számlavezetőn belül','Papíralapon')
            if chosen:
                quote,cells=chosen
                if not outside and 'sajat' not in q and not 'otp-n belul' in q:
                    both=row(h,'Számlavezetőn kívül','Papíralapon')
                    if not both or both[1][-1]!=cells[-1]:return None
                return supported(f'Belföldi eseti forintátutalás papíralapú, fióki benyújtásának díja {cells[-1]}.',[(h,quote)])
    return None


def _simple(question:str,hits:list[Hit]) -> dict | None:
    q=norm(question)
    def give(doc:str,sid:str,words:tuple[str,...], intro:str):
        h=section(hits,doc,sid);line=bullet(h,*words)
        return supported(f'{intro}: {last_fee(line)}.',[(h,line)]) if line and h else None
    if 'mobilbank' in q or 'internetbank' in q:
        h=section(hits,FEE,'SRC-P03-B03');monthly=bullet(h,'havi díja');annual=bullet(h,'teljes éves díja')
        if h and monthly and annual:
            return supported(f'Az OTP MobilBank / InternetBank Alap szolgáltatási csomag díja havonta {last_fee(monthly)}, évente {last_fee(annual)}.',[(h,monthly),(h,annual)])
    if any(x in q for x in ('szamlazar','megszuntetem','szamlavalt')):
        target='Számlaváltás díja' if 'szamlavalt' in q else 'Számlazárás díja'
        return give(FEE,'SRC-P03-B02',(target,),target)
    if any(x in q for x in ('fedezetigazolas', 'egyenlegigazolas')):
        target='Fedezetigazolás' if 'fedezetigazolas' in q else 'Számlaegyenleg-igazolás'
        return give(FEE,'SRC-P09-B03',(target,),target+' díja')
    if 'befizet' in q and ('atm' in q or 'keszpenz' in q):
        if any(x in q for x in ('sajat','otp','otp-s')):
            return give(FEE,'SRC-P08-B03',('Saját ATM-be',),'A belföldi forint készpénzbefizetés díja saját ATM-be')
    if 'kartya' in q and any(x in q for x in ('potlas','potkartya','potol')):
        h=section(hits,NOTICE,'source-p1-card-fees');r=row(h,'pótkártya díja')
        if r:
            quote,cells=r
            return supported(f'A Mastercard Online Diákhitel pótkártya díja {cells[-1]}.',[(h,quote)])
    if 'kartya' in q and ('vasarlas' in q or 'vasarlo' in q):
        return give(FEE,'SRC-P07-B02',('Betéti kártyával',),'A belföldi betéti bankkártyás vásárlás díja')
    if 'hitelkartya' in q:
        h=section(hits,FEE,'SRC-P07-B01')
        if h:
            line=next((ln for ln in h['content'].splitlines() if 'nem áll rendelkezésre' in ln),None)
            if line:return supported('Az OTP Diákhitel Számla díjjegyzéke szerint a hitelkártya-szolgáltatás nem áll rendelkezésre.',[(h,line)])
    if 'limitmodositas' in q or ('atm' in q and 'limit' in q):
        h=section(hits,FEE,'SRC-P09-B02')
        if h:
            quote=next((ln for ln in h['content'].splitlines() if 'Díjtípust nem alkalmazzuk' in ln),None)
            if quote:
                note=next((ln for ln in h['content'].splitlines() if 'nem azonos a 0 Ft-os' in ln),None)
                evidence=[(h,quote)] + ([(h,note)] if note else [])
                return supported('A díjjegyzék az ATM-/vásárlási limit módosításánál „Díjtípust nem alkalmazzuk” jelölést használ, ami nem azonos a 0 Ft-os díjjal.',evidence)
    if 'kartya' in q and any(x in q for x in ('elso eves','masodik eves','masodik evtol','eves dij','elso ev')):
        h=section(hits,FEE,'SRC-P06-B04')
        key='második évtől' if any(x in q for x in ('masodik','evtol')) else 'első éves'
        quote=bullet(h,'Főkártya',key)
        if quote and h:
            return supported(f'A Mastercard Online Diákhitel főkártya {key} díja {last_fee(quote)}.',[(h,quote)])
    if 'csoportos' in q and 'beszed' in q:
        h=section(hits,NOTICE,'source-p1-direct-debit');r=row(h,'Csoportos beszedési megbízás')
        if r:
            quote,cells=r;return supported(f'A csoportos beszedési megbízás díja {cells[-1]}.',[(h,quote)])
    return None


def resolve_fee(question:str,hits:list[Hit]) -> dict | None:
    """Return fully sourced fee/limitation, or None for a non-covered intent."""
    return _atm(question,hits) or _sms(question,hits) or _transfer(question,hits) or _simple(question,hits)
