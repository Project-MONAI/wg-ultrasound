"""Known-positive metadata PHI: canary injection and byte-level leak scanning.

Public data are already de-identified, so privacy cannot be measured on them directly. Every input file
is copied and seeded with synthetic identifiers in seven PHI categories. Each canary carries its own
random 8-character token, so a token found anywhere in an output file is an unambiguous leak of that field.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from .dicom_utils import sstr

PRIVACY_CATS = ["patient_direct", "care_context", "free_text", "nested_sequence", "private_tags", "uids", "dates"]
SEEDED_DATE, SEEDED_STUDY_TIME, SEEDED_ACQ_DT = "20010917", "101500", "20010917101734"
SEEDED_INTERVAL_S = 154.0          # AcquisitionDateTime - (StudyDate + StudyTime)


def tok(seed, *parts, n=8) -> str:
    return hashlib.sha256(("|".join(map(str, parts)) + str(seed)).encode()).hexdigest()[:n].upper()


def canary_uid(seed, *parts) -> str:
    h = hashlib.sha256(("uid|" + "|".join(map(str, parts)) + str(seed)).encode()).hexdigest()[:30]
    return "2.25." + str(int(h, 16))


def _with_bytes(can):
    for c in can:
        c["needle_b"] = c["needle"].encode("ascii", "ignore")
    return can


def seed_canaries(src, dst, case_id, study_key, series_key, seed) -> list[dict]:
    """Copy `src` to `dst` with canaries in all PRIVACY_CATS. PixelData is not touched."""
    ds = dcmread(src)
    can = []

    def put(cat, field, value, needle):
        setattr(ds, field, value)
        can.append({"category": cat, "field": field, "value": value, "needle": needle})

    pk = f"patient|{study_key}"          # patient-level tokens are shared by files of the same study
    for field, code in [("PatientName", "PN"), ("PatientID", "PID"), ("OtherPatientNames", "ON")]:
        t = tok(seed, pk, field)
        put("patient_direct", field, f"CANARY^{code}{t}" if "Name" in field else f"{code}{t}", t)
    put("patient_direct", "PatientBirthDate", "19710203", "19710203")
    t = tok(seed, pk, "addr"); put("patient_direct", "PatientAddress", f"{t} CANARY STREET", t)
    t = tok(seed, pk, "tel"); put("patient_direct", "PatientTelephoneNumbers", f"555{t}", t)

    for field, fmt in [("InstitutionName", "CANARY HOSPITAL {t}"), ("InstitutionAddress", "{t} CANARY AVENUE"),
                       ("ReferringPhysicianName", "CANARY^REF{t}"), ("PerformingPhysicianName", "CANARY^PERF{t}"),
                       ("OperatorsName", "CANARY^OP{t}"), ("StationName", "ST{t}"), ("DeviceSerialNumber", "SN{t}"),
                       ("AccessionNumber", "AC{t}"), ("StudyID", "SI{t}")]:
        t = tok(seed, case_id, field)
        put("care_context", field, fmt.format(t=t), t)

    for field, fmt in [("ImageComments", "Seen for CANARY-{t} in bed 4"), ("AdditionalPatientHistory", "Prior visit CANARY-{t}"),
                       ("PatientComments", "Relative CANARY-{t} present")]:
        t = tok(seed, case_id, field)
        put("free_text", field, fmt.format(t=t), t)

    t1, t2, t3 = tok(seed, pk, "opid"), tok(seed, case_id, "rpid"), tok(seed, case_id, "spsid")
    it = Dataset(); it.PatientID = f"OP{t1}"; it.IssuerOfPatientID = "CANARY"
    ds.OtherPatientIDsSequence = Sequence([it])
    ra = Dataset(); ra.RequestedProcedureID = f"RP{t2}"; ra.ScheduledProcedureStepID = f"SP{t3}"
    ds.RequestAttributesSequence = Sequence([ra])
    can += [{"category": "nested_sequence", "field": "OtherPatientIDsSequence>PatientID", "value": it.PatientID, "needle": t1},
            {"category": "nested_sequence", "field": "RequestAttributesSequence>RequestedProcedureID", "value": ra.RequestedProcedureID, "needle": t2},
            {"category": "nested_sequence", "field": "RequestAttributesSequence>ScheduledProcedureStepID", "value": ra.ScheduledProcedureStepID, "needle": t3}]

    t = tok(seed, case_id, "private")
    ds.private_block(0x0029, "CANARY_VENDOR", create=True).add_new(0x10, "LO", f"PRIV{t}")
    can.append({"category": "private_tags", "field": "(0029,xx10) CANARY_VENDOR", "value": f"PRIV{t}", "needle": t})

    uids = {"StudyInstanceUID": canary_uid(seed, "study", study_key), "SeriesInstanceUID": canary_uid(seed, "series", series_key),
            "SOPInstanceUID": canary_uid(seed, "sop", case_id)}
    if "FrameOfReferenceUID" in ds:
        uids["FrameOfReferenceUID"] = canary_uid(seed, "for", series_key)
    for field, v in uids.items():
        put("uids", field, v, v)
    ds.file_meta.MediaStorageSOPInstanceUID = uids["SOPInstanceUID"]

    for field in ("StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate"):
        setattr(ds, field, SEEDED_DATE)
    ds.StudyTime, ds.SeriesTime, ds.ContentTime = SEEDED_STUDY_TIME, "101600", "101800"
    ds.AcquisitionDateTime = SEEDED_ACQ_DT
    can.append({"category": "dates", "field": "Study/Series/Acquisition/Content Date + AcquisitionDateTime",
                "value": SEEDED_DATE, "needle": SEEDED_DATE})

    ds.PatientIdentityRemoved = "NO"
    for kw in ("DeidentificationMethod", "DeidentificationMethodCodeSequence"):
        if kw in ds:
            delattr(ds, kw)
    ds.save_as(dst, enforce_file_format=True)
    return _with_bytes(can)


def scan_needles(path, canaries) -> dict:
    """{field: found?} - each canary token is searched in the raw bytes of the whole file."""
    data = Path(path).read_bytes()
    return {c["field"]: (c["needle_b"] in data) for c in canaries if c["needle_b"]}


# ---------------------------------------------------------------- MIDI-B derived answer key
PS315_ATTRS = {
    "patient_direct": ["PatientName", "PatientID", "PatientBirthDate", "PatientBirthTime", "PatientAddress",
                       "PatientTelephoneNumbers", "OtherPatientIDs", "OtherPatientNames", "PatientMotherBirthName",
                       "PatientBirthName", "MilitaryRank", "MedicalRecordLocator", "EthnicGroup", "Occupation",
                       "AdditionalPatientHistory", "PatientComments", "IssuerOfPatientID"],
    "care_context": ["InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName", "ReferringPhysicianName",
                     "ReferringPhysicianAddress", "ReferringPhysicianTelephoneNumbers", "PerformingPhysicianName",
                     "NameOfPhysiciansReadingStudy", "OperatorsName", "PhysiciansOfRecord", "RequestingPhysician",
                     "StationName", "DeviceSerialNumber", "AccessionNumber", "StudyID", "RequestedProcedureID",
                     "ScheduledProcedureStepID", "PerformedProcedureStepID", "PersonName", "ContentCreatorName"],
    "free_text": ["ImageComments", "StudyComments"],
    "dates": ["StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate", "AcquisitionDateTime",
              "StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime"],
    "uids": ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID"],
}
_KW2CAT = {kw: cat for cat, kws in PS315_ATTRS.items() for kw in kws}
_GENERIC = {"NONE", "ANONYMOUS", "UNKNOWN", "YES", "NO"}


def derived_answer_key(path) -> list[dict]:
    """Identifying values present in a file with *realistic* PHI (e.g. MIDI-B synthetic data).

    Every value of the DICOM PS3.15 Annex E identifying attributes listed above (recursively, including
    sequences) plus every private-tag string. This is NOT the official MIDI-B answer key.
    """
    ds = dcmread(path, stop_before_pixels=True)
    can, seen = [], set()

    def add(cat, field, v):
        v = sstr(v)
        if len(v) < 4 or v.upper() in _GENERIC or v in seen:
            return
        seen.add(v)
        can.append({"category": cat, "field": field, "value": v, "needle": v})

    def walk(d, nested):
        for el in d:
            if el.VR == "SQ":
                for it in el.value:
                    walk(it, True)
            elif el.tag.is_private and el.VR in ("LO", "SH", "LT", "ST", "UT", "PN", "DA", "DT"):
                add("private_tags", f"private {el.tag}", el.value)
            elif el.keyword in _KW2CAT:
                add("nested_sequence" if nested else _KW2CAT[el.keyword], el.keyword, el.value)

    walk(ds, False)
    return _with_bytes(can)
