import yaml
import json
import os

def cff_to_citation_json(cff_data):
    """Convert CFF data to citation.json format."""
    citation = {
        "title": cff_data.get("title"),
        "authors": [],
        "date": cff_data.get("date-released"),
        "version": cff_data.get("version"),
        "doi": cff_data.get("doi"),
        "url": cff_data.get("url"),
        "abstract": cff_data.get("abstract"),
        "keywords": cff_data.get("keywords", [])
    }
    
    for author in cff_data.get("authors", []):
        author_data = {
            "family": author.get("family-names"),
            "given": author.get("given-names")
        }
        if "orcid" in author:
            author_data["orcid"] = author["orcid"]
        citation["authors"].append(author_data)
        
    return citation

def cff_to_codemeta(cff_data):
    """Convert CFF data to codemeta.json format."""
    codemeta = {
        "@context": "https://doi.org/10.5063/schema/codemeta-2.0",
        "@type": "SoftwareSourceCode",
        "name": cff_data.get("title"),
        "version": cff_data.get("version"),
        "datePublished": cff_data.get("date-released"),
        "author": [],
        "description": cff_data.get("abstract"),
        "keywords": cff_data.get("keywords", []),
        "license": "https://spdx.org/licenses/" + cff_data.get("license", "CC-BY-4.0"),
        "codeRepository": cff_data.get("repository-code")
    }
    
    for author in cff_data.get("authors", []):
        person = {
            "@type": "Person",
            "familyName": author.get("family-names"),
            "givenName": author.get("given-names")
        }
        if "orcid" in author:
            person["@id"] = author["orcid"]
        codemeta["author"].append(person)
        
    if "doi" in cff_data:
        codemeta["identifier"] = cff_data["doi"]
        
    return codemeta

def main():
    if not os.path.exists("CITATION.cff"):
        print("Error: CITATION.cff not found")
        return

    with open("CITATION.cff", "r") as f:
        cff_data = yaml.safe_load(f)

    citation_json = cff_to_citation_json(cff_data)
    with open("citation.json", "w") as f:
        json.dump(citation_json, f, indent=2, default=str)
    print("Created citation.json")

    codemeta_json = cff_to_codemeta(cff_data)
    with open("codemeta.json", "w") as f:
        json.dump(codemeta_json, f, indent=2, default=str)
    print("Created codemeta.json")

    # Generate CITATION.bib
    bibtex = f"""@article{{smawfield2026jwst,
  title={{{cff_data.get('title')}}},
  author={{Smawfield, Matthew Lukin}},
  journal={{Zenodo}},
  year={{{cff_data.get('date-released').year}}},
  doi={{{cff_data.get('doi')}}},
  url={{{cff_data.get('url')}}},
  note={{Preprint v{cff_data.get('version')} ({cff_data.get('commit', 'Kos')})}}
}}"""
    
    with open("CITATION.bib", "w") as f:
        f.write(bibtex)
    print("Created CITATION.bib")

if __name__ == "__main__":
    main()
