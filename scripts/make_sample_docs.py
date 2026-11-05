"""Generate two fictional sample PDFs in docs/ so the app works out of the box.

    python scripts/make_sample_docs.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parents[1] / "docs"

CLUB_RULEBOOK = {
    "title": "Campus Tech Club - Rulebook (Sample Document)",
    "pages": [
        [
            ("h", "Section 1. Membership"),
            ("p", "1.1 Membership is open to all enrolled undergraduate and postgraduate students of the university."),
            ("p", "1.2 The annual membership fee is Rs. 500, payable before 31 August each academic year."),
            ("p", "1.3 Students who join after 1 January pay a reduced fee of Rs. 250 for the remaining year."),
            ("p", "1.4 Membership lapses automatically if the fee is not paid within 30 days of the due date."),
            ("p", "1.5 Alumni may become associate members but cannot vote or hold office."),
            ("h", "Section 2. Meetings"),
            ("p", "2.1 General body meetings are held on the first Friday of every month at 4:00 PM."),
            ("p", "2.2 A quorum for a general body meeting is one third of all paid members."),
            ("p", "2.3 Notice of a general body meeting must be sent at least 72 hours in advance."),
            ("p", "2.4 Minutes of every meeting are published on the club notice board within 7 days."),
        ],
        [
            ("h", "Section 3. Elections"),
            ("p", "3.1 Office bearers are elected every year in the second week of September."),
            ("p", "3.2 Only paid members who have attended at least 3 general body meetings may contest."),
            ("p", "3.3 Voting is by secret ballot, and each member has exactly one vote."),
            ("p", "3.4 A candidate must secure a simple majority of the votes cast to be declared elected."),
            ("p", "3.5 In case of a tie, the faculty advisor casts the deciding vote."),
            ("h", "Section 4. Events and Budget"),
            ("p", "4.1 Every event proposal must be submitted to the Event Lead at least 21 days before the event date."),
            ("p", "4.2 Events with a budget above Rs. 20,000 require written approval from the faculty advisor."),
            ("p", "4.3 Receipts for all expenses must be submitted to the Treasurer within 10 days of the event."),
            ("p", "4.4 Unspent event funds are returned to the club account and cannot be carried over."),
        ],
        [
            ("h", "Section 5. Code of Conduct"),
            ("p", "5.1 Members must treat all participants with respect at club events and online channels."),
            ("p", "5.2 Plagiarism in club hackathons or coding contests leads to immediate disqualification."),
            ("p", "5.3 A first violation of the code of conduct results in a written warning."),
            ("p", "5.4 A second violation results in suspension of membership for one semester."),
            ("p", "5.5 Appeals against a suspension must be filed with the faculty advisor within 14 days."),
            ("h", "Section 6. Amendments"),
            ("p", "6.1 This rulebook may be amended by a two thirds majority of members present and voting."),
            ("p", "6.2 Proposed amendments must be circulated at least 15 days before the vote."),
        ],
    ],
}

COURSE_GUIDELINES = {
    "title": "CS2041 Data Structures - Course Guidelines (Sample Document)",
    "pages": [
        [
            ("h", "1. Course Overview"),
            ("p", "CS2041 is a 4-credit core course with 3 lecture hours and 2 lab hours per week."),
            ("p", "Prerequisite: CS1010 Programming Fundamentals with a minimum grade of C."),
            ("h", "2. Assessment Scheme"),
            ("p", "Continuous Internal Evaluation (CIE) carries 50 marks and the Semester End Exam (SEE) carries 50 marks."),
            ("p", "CIE consists of two internal tests of 15 marks each, a mini project of 15 marks and lab work of 5 marks."),
            ("p", "A student must score at least 40 percent in SEE and 40 percent overall to pass the course."),
            ("p", "Students with less than 75 percent attendance are not permitted to write the SEE."),
            ("h", "3. Mini Project"),
            ("p", "The mini project is done in teams of at most 3 students."),
            ("p", "Topics must be approved by the course instructor by the end of week 4."),
            ("p", "The final report and demo are due in week 13, followed by a 10 minute viva."),
        ],
        [
            ("h", "4. Late Submission and Academic Integrity"),
            ("p", "Lab records submitted late lose 10 percent of the marks per day, up to a maximum of 3 days."),
            ("p", "Submissions more than 3 days late receive zero marks unless a medical certificate is produced."),
            ("p", "Copying code from another student or from the internet without citation is treated as malpractice."),
            ("p", "Malpractice in an internal test results in zero marks for that test."),
            ("h", "5. Syllabus Units"),
            ("p", "Unit 1: Arrays, linked lists and complexity analysis."),
            ("p", "Unit 2: Stacks, queues and their applications."),
            ("p", "Unit 3: Trees, binary search trees and heaps."),
            ("p", "Unit 4: Graphs, BFS, DFS and shortest paths."),
            ("p", "Unit 5: Hashing, sorting and searching techniques."),
            ("h", "6. Textbook"),
            ("p", "Primary text: Data Structures and Algorithm Analysis in C by Mark Allen Weiss, 2nd edition."),
        ],
    ],
}


def build(doc: dict, path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    left, right = 60, width - 60
    for pno, blocks in enumerate(doc["pages"], start=1):
        y = height - 60
        c.setFont("Helvetica-Bold", 14)
        c.drawString(left, y, doc["title"])
        y -= 30
        for kind, text in blocks:
            font, size, gap = ("Helvetica-Bold", 12, 8) if kind == "h" else ("Helvetica", 10.5, 4)
            c.setFont(font, size)
            if kind == "h":
                y -= 6
            for line in simpleSplit(text, font, size, right - left):
                c.drawString(left, y, line)
                y -= size + gap
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(left, 30, f"Fictional sample document for demo purposes - page {pno}")
        c.showPage()
    c.save()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build(CLUB_RULEBOOK, OUT / "sample_club_rulebook.pdf")
    build(COURSE_GUIDELINES, OUT / "sample_course_guidelines.pdf")
    print("Wrote sample PDFs to", OUT)
