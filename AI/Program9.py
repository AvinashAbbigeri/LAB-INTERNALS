from pydantic import BaseModel
from typing import List, Optional
import wikipediaapi, requests, re
from bs4 import BeautifulSoup

class InstitutionProfile(BaseModel):
    founder: Optional[str] = "Unknown"
    Established: Optional[int] = 0
    branches: List[str] = ["Unknown"]
    employee_count: Optional[int] = 0
    summary: str

def fetch_dbpedia_data(name):
    url = f"https://dbpedia.org/data/{name.replace(' ', '_')}.json"
    r = requests.get(url, headers={"User-Agent": "MyInstitutionInfoBot/1.0"})
    if r.status_code != 200: return {}
    j = r.json(); e = f"http://dbpedia.org/resource/{name.replace(' ', '_')}"
    return {"founder": j.get(e, {}).get("http://dbpedia.org/ontology/foundedBy", [{}])[0].get("value", "Unknown")}

def fetch_wikipedia_data(name):
    ua = "MyInstitutionInfoBot/1.0 (contact: your-email@example.com)"
    wiki = wikipediaapi.Wikipedia(user_agent=ua, language="en")
    page = wiki.page(name)
    if not page.exists(): raise ValueError(f"Wikipedia page for '{name}' not found.")
    summary = page.summary[:300]
    url = f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}"
    soup = BeautifulSoup(requests.get(url, headers={"User-Agent": ua}).text, "html.parser")
    infobox = soup.find("table", {"class": "infobox"})
    d = {"founder": "Unknown", "Established": 0, "branches": ["Unknown"], "employee_count": 0, "summary": summary}
    if infobox:
        for row in infobox.find_all("tr"):
            h, v = row.find("th"), row.find("td")
            if h and v:
                k, val = h.text.strip(), v.text.strip()
                if k in ["Founder", "Founders", "Founder(s)"]: d["founder"] = val
                elif k in ["Established", "Founded", "Formation"]:
                    m = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", val)
                    if m: d["Established"] = int(m.group(0))
                elif k in ["Total staff", "Employees", "Staff"]:
                    m = re.search(r"(\d{3,5})", val)
                    if m: d["employee_count"] = int(m.group(0))
                elif k in ["Address", "Location"]: d["branches"] = [val.split(",")[0]]
    if d["founder"] == "Unknown": d.update(fetch_dbpedia_data(name))
    return d

def create_institution_profile(name):
    d = fetch_wikipedia_data(name)
    return InstitutionProfile(**d)

if __name__ == "__main__":
    name = input("Enter institution name: ")
    try:
        print(create_institution_profile(name).model_dump_json(indent=2))
    except ValueError as e:
        print(e)
