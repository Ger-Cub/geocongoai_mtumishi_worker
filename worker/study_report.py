"""Build a downloadable study report from the run and its actual tool trace."""
from base64 import b64decode, b64encode
from html import escape
from io import BytesIO
from re import sub
from urllib.parse import urlsplit


def build_study_report(study, run_id, status, bbox, summary, steps, layers):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer

    buffer = BytesIO()
    title = "Rapport d’étude Gundua"
    doc = SimpleDocTemplate(buffer, title=title, author="GeoCongo AI", pageCompression=1,
                            leftMargin=18 * mm, rightMargin=18 * mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("StudyBody", fontName="Helvetica", fontSize=10, leading=14,
                              alignment=TA_LEFT, spaceAfter=5))
    story = []

    def paragraph(value, style="StudyBody"):
        value = sub(r"(?m)^\s*#{1,6}\s*", "", str(value))
        value = sub(r"\*\*|\x60", "", value)
        # Keep provenance URLs readable and exclude credentials in query fragments.
        value = sub(r"https?://[^\s)]+", lambda m: m[0].split("?")[0].split("#")[0], value)
        story.append(Paragraph(escape(value), styles[style]))

    paragraph(title, "Title")
    paragraph(f"Étude : {study}  |  Exécution : {run_id}")
    paragraph("Statut : " + ("terminée" if status == "completed" else "incomplète ou en échec"))
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        paragraph(f"Emprise EPSG:4326 (ouest, sud, est, nord) : {bbox}")

    paragraph("Synthèse, méthode et limites", "Heading2")
    for line in str(summary or "Aucune synthèse disponible.").splitlines():
        if line.strip():
            paragraph(line, "Heading3" if line.lstrip().startswith("#") else "StudyBody")

    paragraph("Outils exécutés et provenance", "Heading2")
    if not steps:
        paragraph("Aucun résultat d’outil n’a été enregistré.")
    output_names = set()
    for step in steps:
        paragraph(f"{step.get('tool') or 'Outil'} · {step.get('status') or 'état inconnu'}", "Heading3")
        if step.get("error"):
            paragraph(step["error"])
        result = step.get("result")
        if not isinstance(result, dict):
            continue
        metrics = {
            "scene_id": "Scène", "scene_date": "Date d’acquisition", "collection": "Collection",
            "resolution": "Résolution", "coverage_percent": "Couverture comparable (%)",
            "candidate_ha": "Surface candidate (ha)", "candidate_count": "Nombre de zones candidates",
            "area_km2": "Superficie (km²)", "message": "Observation de l’outil",
        }
        for key, label in metrics.items():
            value = result.get(key)
            if isinstance(value, (str, float, int)):
                paragraph(f"{label} : {value}")
        for key in ("scenes_before", "scenes_after", "scenes"):
            scenes = result.get(key)
            if isinstance(scenes, list):
                for scene in scenes[:12]:
                    if not isinstance(scene, dict):
                        continue
                    identifier = scene.get("id") or scene.get("scene_id") or scene.get("item_id")
                    date = scene.get("date") or scene.get("datetime")
                    if identifier:
                        paragraph(f"Scène : {identifier}" + (f" ({date})" if date else ""))
        outputs = result.get("outputs")
        urls = list(outputs.values()) if isinstance(outputs, dict) else []
        urls.extend(result.get(key) for key in ("geojson_url", "raster_url", "geotiff_url", "classified_map_url"))
        for value in urls:
            if isinstance(value, str):
                name = urlsplit(value).path.rsplit("/", 1)[-1]
                if name:
                    output_names.add(name)

    paragraph("Livrables", "Heading2")
    output_names.update(layer["fileName"] for layer in layers if isinstance(layer.get("fileName"), str))
    if output_names:
        for name in sorted(output_names):
            paragraph(name)
    else:
        paragraph("Aucun fichier de données n’a été produit.")
    image = next((layer for layer in layers if layer.get("kind") == "raster_image" and layer.get("contentBase64")), None)
    if image:
        paragraph(image.get("name") or "Aperçu de l’analyse", "Heading3")
        graphic = Image(BytesIO(b64decode(image["contentBase64"])), width=85 * mm, height=85 * mm, kind="proportional")
        story.append(graphic)
        if image.get("legend"):
            paragraph("Légende : " + " · ".join(entry["label"] for entry in image["legend"]))
    story.append(Spacer(1, 5 * mm))

    def footer(canvas, document):
        canvas.setFillColor(colors.HexColor("#246246"))
        canvas.setFont("Helvetica", 9)
        canvas.drawString(18 * mm, 11 * mm, "GeoCongo AI · Mtumishi")
        canvas.drawRightString(document.pagesize[0] - 18 * mm, 11 * mm, str(document.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    data = buffer.getvalue()
    return {"kind": "file", "name": title, "fileName": "rapport-gundua.pdf", "sizeBytes": len(data),
            "contentBase64": b64encode(data).decode("ascii")}
