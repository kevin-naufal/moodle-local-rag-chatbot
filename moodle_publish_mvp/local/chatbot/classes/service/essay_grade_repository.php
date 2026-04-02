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
 * Persistence for essay auto-grading results.
 *
 * @package    local_chatbot
 * @copyright  2026
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class essay_grade_repository {
    /** @var string */
    private const TABLE = 'local_chatbot_essay_grades';

    /**
     * Create one essay grading record.
     *
     * @param int $courseid
     * @param int $graderid
     * @param int $studentid
     * @param int $questionnumber
     * @param string $rubricid
     * @param string $questiontext
     * @param string $expectedkeypoints
     * @param string $studentanswer
     * @param array $grading
     * @param int $assignmentid
     * @param int $cmid
     * @param int $submissionid
     * @param int $attemptnumber
     * @return int
     */
    public function create(
        int $courseid,
        int $graderid,
        int $studentid,
        int $questionnumber,
        string $rubricid,
        string $questiontext,
        string $expectedkeypoints,
        string $studentanswer,
        array $grading,
        int $assignmentid = 0,
        int $cmid = 0,
        int $submissionid = 0,
        int $attemptnumber = 0
    ): int {
        global $DB;

        $now = time();
        $overallscore = isset($grading['overall_score']) ? (float)$grading['overall_score'] : 0.0;
        $confidence = isset($grading['confidence']) ? (float)$grading['confidence'] : 0.0;
        $needsmanualreview = !empty($grading['flags']['needs_manual_review']) ? 1 : 0;

        $record = (object)[
            'courseid' => max(0, $courseid),
            'graderid' => max(0, $graderid),
            'studentid' => max(0, $studentid),
            'question_number' => max(1, $questionnumber),
            'rubric_id' => trim($rubricid) !== '' ? trim($rubricid) : 'essay_default_v1',
            'assignmentid' => $assignmentid > 0 ? $assignmentid : null,
            'cmid' => $cmid > 0 ? $cmid : null,
            'submissionid' => $submissionid > 0 ? $submissionid : null,
            'attemptnumber' => max(0, $attemptnumber),
            'question_text' => trim($questiontext),
            'expected_key_points' => trim($expectedkeypoints),
            'student_answer' => trim($studentanswer),
            'grade_json' => json_encode($grading, JSON_UNESCAPED_UNICODE),
            'overall_score' => $overallscore,
            'confidence' => $confidence,
            'needs_manual_review' => $needsmanualreview,
            'timecreated' => $now,
            'timemodified' => $now,
        ];

        if ($submissionid > 0) {
            $existing = $DB->get_record(
                self::TABLE,
                [
                    'submissionid' => $submissionid,
                    'question_number' => max(1, $questionnumber),
                ],
                'id',
                IGNORE_MISSING
            );
            if ($existing) {
                $updaterecord = clone $record;
                $updaterecord->id = (int)$existing->id;
                unset($updaterecord->timecreated);
                $DB->update_record(self::TABLE, $updaterecord);
                return (int)$updaterecord->id;
            }
        }

        return (int)$DB->insert_record(self::TABLE, $record);
    }
}
