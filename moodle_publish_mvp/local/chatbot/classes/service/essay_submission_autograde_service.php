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
 * Auto-grade essay submissions for assign activities configured by local_chatbot.
 *
 * @package    local_chatbot
 * @copyright  2026
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class essay_submission_autograde_service {
    /** @var string */
    private const CFG_TABLE = 'local_chatbot_essay_autocfg';

    /**
     * Ingest one assign submission and auto-grade when enabled.
     *
     * @param int $submissionid
     * @param int $cmid
     * @param int $courseid
     * @param int $actorid
     * @return void
     */
    public static function ingest_submission(int $submissionid, int $cmid, int $courseid, int $actorid = 0): void {
        global $DB;

        if ($submissionid <= 0 || $cmid <= 0 || $courseid <= 0) {
            return;
        }

        $cfg = $DB->get_record(
            self::CFG_TABLE,
            ['cmid' => $cmid, 'enabled' => 1],
            '*',
            IGNORE_MISSING
        );
        if (!$cfg) {
            return;
        }

        $submission = $DB->get_record(
            'assign_submission',
            ['id' => $submissionid],
            'id,assignment,userid,attemptnumber,status',
            IGNORE_MISSING
        );
        if (!$submission || (int)$submission->userid <= 0) {
            return;
        }
        if ((int)$submission->assignment !== (int)$cfg->assignid) {
            return;
        }
        if (trim((string)$submission->status) !== 'submitted') {
            return;
        }

        $onlinetext = $DB->get_field(
            'assignsubmission_onlinetext',
            'onlinetext',
            [
                'assignment' => (int)$submission->assignment,
                'submission' => (int)$submission->id,
            ],
            IGNORE_MISSING
        );
        $studentanswer = trim((string)html_to_text((string)$onlinetext, 0, false));
        if ($studentanswer === '') {
            return;
        }

        $config = json_decode((string)$cfg->config_json, true);
        if (!is_array($config)) {
            $config = [];
        }

        $questions = [];
        if (isset($config['questions']) && is_array($config['questions'])) {
            $questions = array_values($config['questions']);
        }
        if (empty($questions)) {
            $questions[] = [
                'number' => 1,
                'stem' => trim((string)($config['assignment_title'] ?? 'Essay submission')),
            ];
        }
        $answerkey = isset($config['answer_key']) && is_array($config['answer_key']) ? $config['answer_key'] : [];
        $rubricid = trim((string)($cfg->rubric_id ?? 'essay_default_v1'));
        if ($rubricid === '') {
            $rubricid = 'essay_default_v1';
        }

        $graderid = 0;
        $admin = get_admin();
        if ($admin && !empty($admin->id)) {
            $graderid = (int)$admin->id;
        } else if ($actorid > 0) {
            $graderid = $actorid;
        } else {
            $graderid = (int)$submission->userid;
        }

        $autograder = new essay_autograder();
        $repository = new essay_grade_repository();

        $totalscore = 0.0;
        $gradedcount = 0;
        foreach ($questions as $index => $question) {
            if (!is_array($question)) {
                continue;
            }

            $questionnumber = max(1, (int)($question['number'] ?? ($index + 1)));
            $questiontext = trim((string)($question['stem'] ?? ''));
            if ($questiontext === '') {
                continue;
            }
            $expectedkeypoints = trim((string)($answerkey[(string)$questionnumber] ?? ''));
            if ($expectedkeypoints === '') {
                $expectedkeypoints = trim((string)($answerkey[$questionnumber] ?? ''));
            }
            if ($expectedkeypoints === '') {
                $expectedkeypoints = 'No explicit key points provided.';
            }

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
                (int)$submission->userid,
                $questionnumber,
                $rubricid,
                $questiontext,
                $expectedkeypoints,
                $studentanswer,
                $grading,
                (int)$submission->assignment,
                $cmid,
                (int)$submission->id,
                (int)$submission->attemptnumber
            );

            $totalscore += (float)($grading['overall_score'] ?? 0.0);
            $gradedcount++;
        }

        if ($gradedcount <= 0) {
            return;
        }

        self::apply_assign_grade(
            $courseid,
            $cmid,
            (int)$submission->assignment,
            (int)$submission->userid,
            (int)$submission->attemptnumber,
            $graderid,
            $totalscore / $gradedcount
        );
    }

    /**
     * Apply calculated score to Moodle assignment gradebook.
     *
     * @param int $courseid
     * @param int $cmid
     * @param int $assignid
     * @param int $userid
     * @param int $attemptnumber
     * @param int $graderid
     * @param float $score100
     * @return void
     */
    private static function apply_assign_grade(
        int $courseid,
        int $cmid,
        int $assignid,
        int $userid,
        int $attemptnumber,
        int $graderid,
        float $score100
    ): void {
        global $CFG;

        if ($courseid <= 0 || $cmid <= 0 || $assignid <= 0 || $userid <= 0) {
            return;
        }

        require_once($CFG->dirroot . '/mod/assign/locallib.php');

        $cm = get_coursemodule_from_id('assign', $cmid, $courseid, false, IGNORE_MISSING);
        if (!$cm || (int)$cm->instance !== $assignid) {
            return;
        }

        $course = get_course($courseid);
        $context = \context_module::instance((int)$cm->id);
        $assign = new \assign($context, $cm, $course);
        $instance = $assign->get_instance();
        $maxgrade = (float)($instance->grade ?? 0.0);
        if ($maxgrade <= 0.0) {
            return;
        }

        $grade = $assign->get_user_grade($userid, true, $attemptnumber);
        if (!$grade || !isset($grade->id)) {
            return;
        }

        $normalized = max(0.0, min(100.0, $score100));
        $converted = round(($normalized / 100.0) * $maxgrade, 2);

        $grade->grade = $converted;
        $grade->grader = $graderid > 0 ? $graderid : -1;
        $grade->attemptnumber = $attemptnumber;
        $assign->update_grade($grade, false);
    }
}
