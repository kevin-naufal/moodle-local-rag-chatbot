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
 * Auto-grade essay quiz attempts when teacher enables essay auto-grade toggle.
 *
 * @package    local_chatbot
 * @copyright  2026
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class essay_quiz_autograde_service {
    /** @var string */
    private const CFG_TABLE = 'local_chatbot_essay_autocfg';

    /**
     * Ingest one quiz attempt and apply essay auto-grade if configured.
     *
     * @param int $attemptid
     * @param int $cmid
     * @param int $courseid
     * @param int $actorid
     * @return void
     */
    public static function ingest_attempt(int $attemptid, int $cmid, int $courseid, int $actorid = 0): void {
        global $CFG, $DB;

        if ($attemptid <= 0 || $cmid <= 0 || $courseid <= 0) {
            return;
        }

        $dbman = $DB->get_manager();
        if (!$dbman->table_exists(new \xmldb_table(self::CFG_TABLE))) {
            return;
        }

        $cfg = $DB->get_record(self::CFG_TABLE, ['cmid' => $cmid, 'enabled' => 1], '*', IGNORE_MISSING);
        if (!$cfg) {
            return;
        }

        $attempt = $DB->get_record(
            'quiz_attempts',
            ['id' => $attemptid],
            'id,quiz,userid,attempt,uniqueid,state',
            IGNORE_MISSING
        );
        if (!$attempt || (int)$attempt->quiz <= 0 || (int)$attempt->userid <= 0) {
            return;
        }
        if ((int)$cfg->assignid > 0 && (int)$cfg->assignid !== (int)$attempt->quiz) {
            return;
        }
        if ((string)$attempt->state !== 'finished') {
            return;
        }

        require_once($CFG->dirroot . '/mod/quiz/locallib.php');
        require_once($CFG->dirroot . '/question/engine/lib.php');

        $quiz = $DB->get_record('quiz', ['id' => (int)$attempt->quiz], '*', IGNORE_MISSING);
        if (!$quiz) {
            return;
        }

        $config = json_decode((string)$cfg->config_json, true);
        if (!is_array($config)) {
            $config = [];
        }

        $questions = isset($config['questions']) && is_array($config['questions']) ? array_values($config['questions']) : [];
        $answerkey = isset($config['answer_key']) && is_array($config['answer_key']) ? $config['answer_key'] : [];
        $rubricid = trim((string)($cfg->rubric_id ?? 'essay_default_v1'));
        if ($rubricid === '') {
            $rubricid = 'essay_default_v1';
        }

        $autograder = new essay_autograder();
        $repository = new essay_grade_repository();
        $graderid = self::resolve_grader_id($actorid, (int)$attempt->userid);

        $quba = \question_engine::load_questions_usage_by_activity((int)$attempt->uniqueid);
        $gradedcount = 0;
        foreach ($quba->get_slots() as $slot) {
            $qa = $quba->get_question_attempt((int)$slot);
            $question = $qa->get_question(false);
            $questiontype = '';
            if ($question && method_exists($question, 'get_type_name')) {
                $questiontype = (string)$question->get_type_name();
            }
            if (!$question || $questiontype !== 'essay') {
                continue;
            }

            $questionnumber = max(1, (int)$slot);
            $questiontext = self::resolve_question_text($questions, $questionnumber, (string)$qa->get_question_summary());
            if ($questiontext === '') {
                continue;
            }

            $expectedkeypoints = self::resolve_expected_key_points($answerkey, $questionnumber);
            $studentanswer = self::extract_student_answer($qa);

            $grading = $autograder->grade([
                'question_text' => $questiontext,
                'expected_key_points' => $expectedkeypoints,
                'student_answer' => $studentanswer,
                'question_number' => $questionnumber,
                'rubric_id' => $rubricid,
            ]);

            $repository->create(
                $courseid,
                $graderid,
                (int)$attempt->userid,
                $questionnumber,
                $rubricid,
                $questiontext,
                $expectedkeypoints,
                $studentanswer,
                $grading,
                (int)$attempt->quiz,
                $cmid,
                (int)$attempt->id,
                (int)$attempt->attempt
            );

            $maxmark = max(0.0, (float)$qa->get_max_mark());
            if ($maxmark <= 0.0) {
                $maxmark = 1.0;
            }
            $score100 = max(0.0, min(100.0, (float)($grading['overall_score'] ?? 0.0)));
            $mark = round(($score100 / 100.0) * $maxmark, 5);
            $comment = self::build_question_feedback_comment($grading);
            $quba->manual_grade((int)$slot, $comment, $mark, FORMAT_HTML);
            $gradedcount++;
        }

        if ($gradedcount <= 0) {
            return;
        }

        \question_engine::save_questions_usage_by_activity($quba);
        $sumgrades = round((float)$quba->get_total_mark(), 5);
        $DB->set_field('quiz_attempts', 'sumgrades', $sumgrades, ['id' => (int)$attempt->id]);
        quiz_save_best_grade($quiz, (int)$attempt->userid);
    }

    /**
     * Build concise per-question feedback HTML from grading payload.
     *
     * @param array $grading
     * @return string
     */
    private static function build_question_feedback_comment(array $grading): string {
        $parts = [];
        $score = (float)($grading['overall_score'] ?? 0.0);
        $max = (float)($grading['max_score'] ?? 100.0);
        $parts[] = '<p><strong>Auto-grading rationale</strong></p>';
        $parts[] = '<p>Score: ' . format_float($score, 2) . ' / ' . format_float($max, 0) . '</p>';

        $criteria = $grading['criterion_scores'] ?? [];
        if (is_array($criteria) && !empty($criteria)) {
            $rows = [];
            foreach ($criteria as $item) {
                if (!is_array($item)) {
                    continue;
                }
                $key = trim((string)($item['criterion_key'] ?? 'criterion'));
                $level = (int)($item['level'] ?? 0);
                $cscore = (float)($item['score'] ?? 0.0);
                $reason = trim((string)($item['reason'] ?? ''));
                $label = str_replace('_', ' ', $key);
                $label = ucwords($label);
                $line = $label . ': level ' . $level . ', score ' . format_float($cscore, 2);
                if ($reason !== '') {
                    $line .= ' - ' . $reason;
                }
                $rows[] = '<li>' . s($line) . '</li>';
            }
            if (!empty($rows)) {
                $parts[] = '<ul>' . implode('', $rows) . '</ul>';
            }
        }

        $improvements = $grading['improvement_suggestions'] ?? [];
        if (is_array($improvements) && !empty($improvements)) {
            $rows = [];
            foreach (array_slice($improvements, 0, 3) as $tip) {
                $tip = trim((string)$tip);
                if ($tip !== '') {
                    $rows[] = '<li>' . s($tip) . '</li>';
                }
            }
            if (!empty($rows)) {
                $parts[] = '<p><strong>Improvement suggestions</strong></p>';
                $parts[] = '<ul>' . implode('', $rows) . '</ul>';
            }
        }

        $missing = $grading['missing_key_points'] ?? [];
        if (is_array($missing) && !empty($missing)) {
            $rows = [];
            foreach (array_slice($missing, 0, 3) as $m) {
                $m = trim((string)$m);
                if ($m !== '') {
                    $rows[] = '<li>' . s($m) . '</li>';
                }
            }
            if (!empty($rows)) {
                $parts[] = '<p><strong>Missing key points</strong></p>';
                $parts[] = '<ul>' . implode('', $rows) . '</ul>';
            }
        }

        $flags = $grading['flags']['reasons'] ?? [];
        if (is_array($flags) && !empty($flags)) {
            $rows = [];
            foreach (array_slice($flags, 0, 3) as $reason) {
                $reason = trim((string)$reason);
                if ($reason !== '') {
                    $rows[] = '<li>' . s($reason) . '</li>';
                }
            }
            if (!empty($rows)) {
                $parts[] = '<p><strong>Manual review flags</strong></p>';
                $parts[] = '<ul>' . implode('', $rows) . '</ul>';
            }
        }

        return implode("\n", $parts);
    }

    /**
     * Resolve essay question text from config with fallback from question summary.
     *
     * @param array $questions
     * @param int $questionnumber
     * @param string $fallback
     * @return string
     */
    private static function resolve_question_text(array $questions, int $questionnumber, string $fallback): string {
        $index = max(0, $questionnumber - 1);
        if (isset($questions[$index]) && is_array($questions[$index])) {
            $stem = trim((string)($questions[$index]['stem'] ?? ''));
            if ($stem !== '') {
                return $stem;
            }
        }

        $normalizedfallback = trim((string)html_to_text($fallback, 0, false));
        if ($normalizedfallback !== '') {
            return $normalizedfallback;
        }

        return trim((string)$fallback);
    }

    /**
     * Resolve expected key points from answer-key map.
     *
     * @param array $answerkey
     * @param int $questionnumber
     * @return string
     */
    private static function resolve_expected_key_points(array $answerkey, int $questionnumber): string {
        $value = '';
        if (array_key_exists((string)$questionnumber, $answerkey)) {
            $value = trim((string)$answerkey[(string)$questionnumber]);
        } else if (array_key_exists($questionnumber, $answerkey)) {
            $value = trim((string)$answerkey[$questionnumber]);
        }

        if ($value === '') {
            $value = 'No explicit key points provided.';
        }
        return $value;
    }

    /**
     * Extract student answer text from one essay question attempt.
     *
     * @param \question_attempt $qa
     * @return string
     */
    private static function extract_student_answer(\question_attempt $qa): string {
        $data = $qa->get_last_qt_data();
        $raw = '';
        foreach (['answer', 'answertext', 'response'] as $key) {
            if (isset($data[$key])) {
                $raw = (string)$data[$key];
                if (trim($raw) !== '') {
                    break;
                }
            }
        }

        if (trim($raw) === '') {
            $raw = (string)$qa->get_response_summary();
        }

        $plaintext = trim((string)html_to_text($raw, 0, false));
        if ($plaintext !== '') {
            return $plaintext;
        }

        return trim(strip_tags((string)$raw));
    }

    /**
     * Resolve grader user id for stored grading records.
     *
     * @param int $actorid
     * @param int $userid
     * @return int
     */
    private static function resolve_grader_id(int $actorid, int $userid): int {
        $admin = get_admin();
        if ($admin && !empty($admin->id)) {
            return (int)$admin->id;
        }
        if ($actorid > 0) {
            return $actorid;
        }
        return max(0, $userid);
    }
}
