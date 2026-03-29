<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Moodle is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Moodle.  If not, see <http://www.gnu.org/licenses/>.

namespace local_chatbot\service;

defined('MOODLE_INTERNAL') || die();

/**
 * Publishes a draft into Moodle course as mod_assign.
 *
 * @package    local_chatbot
 * @copyright  2026
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class publisher {
    /**
     * Publish a draft and return module info.
     *
     * @param \stdClass $draft
     * @param \stdClass $course
     * @return array{cmid:int,modulename:string}
     */
    public function publish(\stdClass $draft, \stdClass $course): array {
        $payload = json_decode((string)$draft->draft_json, true);
        $contentmode = $this->normalize_content_mode(
            is_array($payload) ? (string)($payload['content_mode'] ?? 'assignment') : 'assignment'
        );
        $assignmenttype = $this->normalize_assignment_type((string)($draft->assignment_type ?? ''));
        if ($assignmenttype === 'multiple-choice') {
            $cmid = $this->publish_quiz($draft, $course, $contentmode === 'practice');
            return ['cmid' => $cmid, 'modulename' => 'quiz'];
        }

        $cmid = $this->publish_assign($draft, $course);
        return ['cmid' => $cmid, 'modulename' => 'assign'];
    }

    /**
     * Publish one draft as Assignment activity.
     *
     * @param \stdClass $draft
     * @param \stdClass $course
     * @return int created course module id
     */
    public function publish_assign(\stdClass $draft, \stdClass $course): int {
        global $CFG, $DB;

        require_once($CFG->dirroot . '/course/modlib.php');

        $payload = json_decode((string)$draft->draft_json, true);
        if (!is_array($payload)) {
            throw new \moodle_exception('invaliddraftjson', 'local_chatbot');
        }

        $module = $DB->get_record('modules', ['name' => 'assign'], '*', MUST_EXIST);
        $assignmenttype = $this->normalize_assignment_type((string)($draft->assignment_type ?? ''));
        $intro = $this->build_student_intro($payload, $assignmenttype, 'assignment');
        $section = $this->resolve_section_by_topic($course, (string)($payload['topic'] ?? ''));

        $item = (object)[
            'course' => (int)$course->id,
            'module' => (int)$module->id,
            'modulename' => 'assign',
            'section' => $section,
            'name' => trim((string)$payload['assignment_title']),
            'intro' => $intro,
            'introformat' => FORMAT_HTML,
            'visible' => 1,
            'alwaysshowdescription' => 1,
            'submissiondrafts' => 0,
            'requiresubmissionstatement' => 0,
            'sendnotifications' => 0,
            'sendlatenotifications' => 0,
            'sendstudentnotifications' => 0,
            'duedate' => 0,
            'allowsubmissionsfromdate' => 0,
            'cutoffdate' => 0,
            'gradingduedate' => 0,
            'grade' => 100,
            'assignsubmission_onlinetext_enabled' => 1,
            'assignsubmission_file_enabled' => 0,
            'assignfeedback_comments_enabled' => 1,
            'attemptreopenmethod' => 'none',
            'maxattempts' => -1,
            'teamsubmission' => 0,
            'requireallteammemberssubmit' => 0,
            'blindmarking' => 0,
            'markingworkflow' => 0,
            'markingallocation' => 0,
            'completionsubmit' => 0,
        ];

        $result = add_moduleinfo($item, $course);
        if (empty($result->coursemodule)) {
            throw new \moodle_exception('publishfailed', 'local_chatbot');
        }

        return (int)$result->coursemodule;
    }

    /**
     * Publish one draft as Quiz activity with auto-graded multichoice questions.
     *
     * @param \stdClass $draft
     * @param \stdClass $course
     * @return int created course module id
     */
    public function publish_quiz(\stdClass $draft, \stdClass $course, bool $ispractice = false): int {
        global $CFG, $DB, $USER;

        require_once($CFG->dirroot . '/course/modlib.php');
        require_once($CFG->dirroot . '/mod/quiz/lib.php');
        require_once($CFG->dirroot . '/mod/quiz/locallib.php');
        require_once($CFG->libdir . '/questionlib.php');

        $payload = json_decode((string)$draft->draft_json, true);
        if (!is_array($payload)) {
            throw new \moodle_exception('invaliddraftjson', 'local_chatbot');
        }

        $module = $DB->get_record('modules', ['name' => 'quiz'], '*', MUST_EXIST);
        $intro = $this->build_student_intro($payload, 'multiple-choice', $ispractice ? 'practice' : 'assignment');
        $section = $this->resolve_section_by_topic($course, (string)($payload['topic'] ?? ''));
        $title = trim((string)$payload['assignment_title']);
        if ($ispractice && stripos($title, '[Practice]') !== 0) {
            $title = '[Practice] ' . $title;
        }

        $item = (object)[
            'course' => (int)$course->id,
            'module' => (int)$module->id,
            'modulename' => 'quiz',
            'section' => $section,
            'name' => $title,
            'intro' => $intro,
            'introformat' => FORMAT_HTML,
            'visible' => 1,
            'timeopen' => 0,
            'timeclose' => 0,
            'timelimit' => 0,
            'overduehandling' => 'autoabandon',
            'graceperiod' => 0,
            'preferredbehaviour' => $ispractice ? 'immediatefeedback' : 'deferredfeedback',
            'canredoquestions' => $ispractice ? 1 : 0,
            'attempts' => $ispractice ? 0 : 1,
            'attemptonlast' => 0,
            'grademethod' => $ispractice ? 1 : 1,
            'decimalpoints' => 2,
            'questiondecimalpoints' => -1,
            'questionsperpage' => 1,
            'navmethod' => 'free',
            'shuffleanswers' => 1,
            'grade' => 100,
            'quizpassword' => '',
            'subnet' => '',
            'browsersecurity' => '-',
            'delay1' => 0,
            'delay2' => 0,
            'showuserpicture' => 0,
            'showblocks' => 0,
            // Enable post-attempt review so students can open review page and see grades.
            'attemptduring' => 1,
            'attemptimmediately' => 1,
            'attemptopen' => 1,
            'attemptclosed' => 1,
            'correctnessduring' => 1,
            'correctnessimmediately' => 1,
            'correctnessopen' => 1,
            'correctnessclosed' => 1,
            'marksduring' => 1,
            'marksimmediately' => 1,
            'marksopen' => 1,
            'marksclosed' => 1,
            'specificfeedbackduring' => 1,
            'specificfeedbackimmediately' => 1,
            'specificfeedbackopen' => 1,
            'specificfeedbackclosed' => 1,
            'generalfeedbackduring' => 1,
            'generalfeedbackimmediately' => 1,
            'generalfeedbackopen' => 1,
            'generalfeedbackclosed' => 1,
            'rightanswerduring' => 1,
            'rightanswerimmediately' => 1,
            'rightansweropen' => 1,
            'rightanswerclosed' => 1,
            'overallfeedbackimmediately' => 1,
            'overallfeedbackopen' => 1,
            'overallfeedbackclosed' => 1,
            'feedbacktext' => [
                ['text' => '', 'format' => FORMAT_HTML, 'itemid' => 0],
            ],
            'feedbackboundaries' => [],
            'completionpassgrade' => 0,
            'completionattemptsexhausted' => 0,
            'completionminattemptsenabled' => 0,
            'completionminattempts' => 0,
        ];

        $result = add_moduleinfo($item, $course);
        if (empty($result->coursemodule) || empty($result->instance)) {
            throw new \moodle_exception('publishfailed', 'local_chatbot');
        }

        $quiz = $DB->get_record('quiz', ['id' => (int)$result->instance], '*', MUST_EXIST);
        $coursecontext = \context_course::instance((int)$course->id);
        $qcategory = question_get_default_category($coursecontext->id);
        if (!$qcategory) {
            $qcategory = question_make_default_categories([$coursecontext]);
        }

        $questions = (array)($payload['questions'] ?? []);
        $answerkey = (array)($payload['answer_key'] ?? []);
        $slot = 1;
        foreach ($questions as $questiondata) {
            if (!is_array($questiondata)) {
                continue;
            }
            $questionid = $this->create_multichoice_question(
                $questiondata,
                (string)($answerkey[(string)$slot] ?? ''),
                (int)$qcategory->id,
                (int)$USER->id
            );
            quiz_add_quiz_question($questionid, $quiz, 0, 1.0);
            $slot++;
        }

        quiz_update_sumgrades($quiz);
        return (int)$result->coursemodule;
    }

    /**
     * Build student-facing intro from draft sections.
     * Answer key is intentionally excluded.
     *
     * @param array $payload
     * @return string
     */
    private function build_student_intro(array $payload, string $assignmenttype = 'essay', string $contentmode = 'assignment'): string {
        $parts = [];

        if ($contentmode === 'practice') {
            $parts[] = '<p><strong>Practice Quiz</strong> - This activity is for self-practice and immediate feedback.</p>';
        }

        $parts[] = '<h3>Learning Objectives</h3>';
        $parts[] = $this->to_html_list((array)($payload['learning_objectives'] ?? []), true);

        $instructions = trim((string)($payload['instructions'] ?? ''));
        if ($instructions !== '') {
            $parts[] = '<h3>Instructions</h3>';
            $parts[] = '<p>' . s($instructions) . '</p>';
        }

        if ($assignmenttype !== 'multiple-choice') {
            $parts[] = '<h3>Question List</h3>';
            $parts[] = $this->to_question_list_stem_only((array)($payload['questions'] ?? []));
        }

        $parts[] = '<h3>Grading Rubric</h3>';
        $parts[] = $this->to_html_list((array)($payload['grading_rubric'] ?? []), false);

        return implode("\n", $parts);
    }

    /**
     * Render plain list as HTML list.
     *
     * @param array $items
     * @param bool $ordered
     * @return string
     */
    private function to_html_list(array $items, bool $ordered = false): string {
        if (empty($items)) {
            return '<p>-</p>';
        }
        $tag = $ordered ? 'ol' : 'ul';
        $html = '<' . $tag . '>';
        foreach ($items as $item) {
            $html .= '<li>' . s((string)$item) . '</li>';
        }
        $html .= '</' . $tag . '>';
        return $html;
    }

    /**
     * Resolve course section number from selected topic name.
     *
     * @param \stdClass $course
     * @param string $topic
     * @return int
     */
    private function resolve_section_by_topic(\stdClass $course, string $topic): int {
        global $DB;

        $normalizedtopic = $this->normalize_topic_label($topic);
        if ($normalizedtopic === '') {
            return 0;
        }

        $sections = $DB->get_records(
            'course_sections',
            ['course' => (int)$course->id],
            'section ASC',
            'section,name'
        );

        foreach ($sections as $section) {
            $label = trim((string)$section->name);
            if ($label === '') {
                $label = 'Topic ' . (int)$section->section;
            }
            if ($this->normalize_topic_label($label) === $normalizedtopic) {
                return (int)$section->section;
            }
        }

        return 0;
    }

    /**
     * Normalize topic label for case-insensitive matching.
     *
     * @param string $label
     * @return string
     */
    private function normalize_topic_label(string $label): string {
        $text = html_entity_decode((string)$label, ENT_QUOTES | ENT_HTML5, 'UTF-8');
        $text = strip_tags($text);
        $text = preg_replace('/\s+/', ' ', trim($text));
        if (!is_string($text)) {
            return '';
        }
        return \core_text::strtolower($text);
    }

    /**
     * Render questions and A-D options.
     *
     * @param array $questions
     * @return string
     */
    private function to_question_list_with_options(array $questions): string {
        if (empty($questions)) {
            return '<p>-</p>';
        }

        $html = '<ol>';
        foreach ($questions as $question) {
            $stem = s((string)($question['stem'] ?? ''));
            $options = (array)($question['options'] ?? []);

            $html .= '<li>';
            $html .= '<p>' . $stem . '</p>';
            $html .= '<ul>';
            foreach (['A', 'B', 'C', 'D'] as $label) {
                $optiontext = s((string)($options[$label] ?? ''));
                $html .= '<li><strong>' . $label . '.</strong> ' . $optiontext . '</li>';
            }
            $html .= '</ul>';
            $html .= '</li>';
        }
        $html .= '</ol>';

        return $html;
    }

    /**
     * Render questions as numbered prompts only (essay/case-study).
     *
     * @param array $questions
     * @return string
     */
    private function to_question_list_stem_only(array $questions): string {
        if (empty($questions)) {
            return '<p>-</p>';
        }

        $html = '<ol>';
        foreach ($questions as $question) {
            $stem = s((string)($question['stem'] ?? ''));
            $html .= '<li><p>' . $stem . '</p></li>';
        }
        $html .= '</ol>';
        return $html;
    }

    /**
     * Normalize assignment type naming variants.
     *
     * @param string $assignmenttype
     * @return string
     */
    private function normalize_assignment_type(string $assignmenttype): string {
        $normalized = strtolower(trim($assignmenttype));
        if ($normalized === 'multiple_choice') {
            return 'multiple-choice';
        }
        if ($normalized === 'multiple-choice') {
            return 'multiple-choice';
        }
        if ($normalized === 'case-study' || $normalized === 'case_study') {
            return 'case-study';
        }
        return 'essay';
    }

    /**
     * Normalize content mode variants.
     *
     * @param string $contentmode
     * @return string
     */
    private function normalize_content_mode(string $contentmode): string {
        $normalized = strtolower(trim($contentmode));
        if ($normalized === 'practice') {
            return 'practice';
        }
        return 'assignment';
    }

    /**
     * Create one multichoice question in question bank and return question id.
     *
     * @param array $questiondata
     * @param string $correctletter
     * @param int $categoryid
     * @param int $userid
     * @return int
     */
    private function create_multichoice_question(array $questiondata, string $correctletter, int $categoryid, int $userid): int {
        $stem = trim((string)($questiondata['stem'] ?? ''));
        if ($stem === '') {
            throw new \moodle_exception('invalidquestionstem', 'local_chatbot');
        }

        $options = (array)($questiondata['options'] ?? []);
        $correct = strtoupper(trim($correctletter));
        if (!in_array($correct, ['A', 'B', 'C', 'D'], true)) {
            throw new \moodle_exception('invalidanswerkeyletter', 'local_chatbot');
        }

        $q = new \stdClass();
        $q->id = 0;
        $q->qtype = 'multichoice';
        $q->createdby = $userid;
        $q->modifiedby = $userid;
        $q->idnumber = null;
        $q->status = \core_question\local\bank\question_version_status::QUESTION_STATUS_READY;

        $fromform = new \stdClass();
        $fromform->category = (string)$categoryid;
        $fromform->name = shorten_text(strip_tags($stem), 120);
        if (trim((string)$fromform->name) === '') {
            $fromform->name = 'Question';
        }
        $fromform->questiontext = ['text' => $stem, 'format' => FORMAT_HTML, 'itemid' => 0];
        $fromform->generalfeedback = ['text' => '', 'format' => FORMAT_HTML, 'itemid' => 0];
        $fromform->defaultmark = 1;
        $fromform->penalty = 0.3333333;
        $fromform->single = 1;
        $fromform->shuffleanswers = 1;
        $fromform->answernumbering = 'abc';
        $fromform->showstandardinstruction = 1;
        $fromform->correctfeedback = ['text' => '', 'format' => FORMAT_HTML, 'itemid' => 0];
        $fromform->partiallycorrectfeedback = ['text' => '', 'format' => FORMAT_HTML, 'itemid' => 0];
        $fromform->incorrectfeedback = ['text' => '', 'format' => FORMAT_HTML, 'itemid' => 0];
        $fromform->hint = [];
        $fromform->hintclearwrong = [];
        $fromform->hintshownumcorrect = [];
        $fromform->status = \core_question\local\bank\question_version_status::QUESTION_STATUS_READY;

        $fromform->answer = [];
        $fromform->fraction = [];
        $fromform->feedback = [];
        foreach (['A', 'B', 'C', 'D'] as $label) {
            $optiontext = trim((string)($options[$label] ?? ''));
            if ($optiontext === '') {
                throw new \moodle_exception('invalidquestionoptions', 'local_chatbot');
            }
            $fromform->answer[] = ['text' => $optiontext, 'format' => FORMAT_HTML];
            $fromform->fraction[] = ($label === $correct) ? '1.0' : '0.0';
            $fromform->feedback[] = ['text' => '', 'format' => FORMAT_HTML, 'itemid' => 0];
        }
        // Keep one trailing empty row to mimic editing form layout.
        $fromform->answer[] = ['text' => '', 'format' => FORMAT_HTML];
        $fromform->fraction[] = '0.0';
        $fromform->feedback[] = ['text' => '', 'format' => FORMAT_HTML, 'itemid' => 0];

        $saved = \question_bank::get_qtype('multichoice')->save_question($q, $fromform);
        return (int)$saved->id;
    }
}
