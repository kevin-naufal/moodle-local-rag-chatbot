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
 * Validates draft payload before publish.
 *
 * @package    local_chatbot
 * @copyright  2026
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class draft_validator {
    /**
     * Validate draft structure and multiple-choice consistency.
     *
     * @param \stdClass $draft
     * @return array decoded payload for downstream use
     */
    public function validate_for_publish(\stdClass $draft): array {
        $payload = json_decode((string)$draft->draft_json, true);
        if (!is_array($payload)) {
            throw new \moodle_exception('invaliddraftjson', 'local_chatbot');
        }
        $this->validate_payload($payload, (int)$draft->question_count, (string)($draft->assignment_type ?? ''));
        return $payload;
    }

    /**
     * Validate payload structure for save/publish.
     *
     * @param array $payload
     * @param int $expectedcount
     * @param string $assignmenttype
     * @return void
     */
    public function validate_payload(array $payload, int $expectedcount = 0, string $assignmenttype = ''): void {
        $expectedcount = max(0, $expectedcount);
        $normalizedtype = $this->normalize_assignment_type($assignmenttype);

        $required = [
            'assignment_title',
            'learning_objectives',
            'instructions',
            'questions',
            'answer_key',
            'grading_rubric',
        ];
        foreach ($required as $key) {
            if (!array_key_exists($key, $payload)) {
                throw new \moodle_exception('missingdraftsection', 'local_chatbot', '', $key);
            }
        }

        if (!is_string($payload['assignment_title']) || trim($payload['assignment_title']) === '') {
            throw new \moodle_exception('invalidassignmenttitle', 'local_chatbot');
        }
        if (!is_array($payload['questions']) || empty($payload['questions'])) {
            throw new \moodle_exception('invalidquestions', 'local_chatbot');
        }

        if ($expectedcount === 0) {
            $expectedcount = count($payload['questions']);
        }
        if ($expectedcount > 0 && count($payload['questions']) !== $expectedcount) {
            throw new \moodle_exception('questioncountmismatch', 'local_chatbot');
        }

        $this->validate_question_stems($payload['questions']);
        if ($normalizedtype === 'multiple-choice') {
            $this->validate_multiple_choice_options($payload['questions']);
            $this->validate_multiple_choice_answer_key($payload['answer_key'], count($payload['questions']));
        } else {
            $this->validate_open_answer_key($payload['answer_key'], count($payload['questions']));
        }
    }

    /**
     * Ensure each question has a non-empty stem.
     *
     * @param array $questions
     * @return void
     */
    private function validate_question_stems(array $questions): void {
        $index = 0;
        foreach ($questions as $question) {
            $index++;
            if (!is_array($question) || empty(trim((string)($question['stem'] ?? '')))) {
                throw new \moodle_exception('invalidquestionstem', 'local_chatbot', '', $index);
            }
        }
    }

    /**
     * Ensure each multiple-choice question has A-D options.
     *
     * @param array $questions
     * @return void
     */
    private function validate_multiple_choice_options(array $questions): void {
        $index = 0;
        foreach ($questions as $question) {
            $index++;
            $options = $question['options'] ?? null;
            if (!is_array($options)) {
                throw new \moodle_exception('invalidquestionoptions', 'local_chatbot', '', $index);
            }

            foreach (['A', 'B', 'C', 'D'] as $option) {
                if (!array_key_exists($option, $options) || trim((string)$options[$option]) === '') {
                    throw new \moodle_exception('invalidquestionoptions', 'local_chatbot', '', $index);
                }
            }
        }
    }

    /**
     * Ensure answer key format is exactly 1..N mapped to A/B/C/D.
     *
     * @param mixed $answerkey
     * @param int $count
     * @return void
     */
    private function validate_multiple_choice_answer_key($answerkey, int $count): void {
        if (!is_array($answerkey) || $count <= 0) {
            throw new \moodle_exception('invalidanswerkey', 'local_chatbot');
        }

        for ($i = 1; $i <= $count; $i++) {
            $key = (string)$i;
            if (!array_key_exists($key, $answerkey)) {
                throw new \moodle_exception('invalidanswerkeynumber', 'local_chatbot', '', $i);
            }

            $value = strtoupper(trim((string)$answerkey[$key]));
            if (!in_array($value, ['A', 'B', 'C', 'D'], true)) {
                throw new \moodle_exception('invalidanswerkeyletter', 'local_chatbot', '', $i);
            }
        }
    }

    /**
     * Ensure essay/case answer key has numbered entries with non-empty key points.
     *
     * @param mixed $answerkey
     * @param int $count
     * @return void
     */
    private function validate_open_answer_key($answerkey, int $count): void {
        if (!is_array($answerkey) || $count <= 0) {
            throw new \moodle_exception('invalidanswerkey', 'local_chatbot');
        }

        for ($i = 1; $i <= $count; $i++) {
            $key = (string)$i;
            if (!array_key_exists($key, $answerkey)) {
                throw new \moodle_exception('invalidanswerkeynumber', 'local_chatbot', '', $i);
            }
            if (trim((string)$answerkey[$key]) === '') {
                throw new \moodle_exception('invalidanswerkey', 'local_chatbot');
            }
        }
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
        if ($normalized === 'essay') {
            return 'essay';
        }
        return 'essay';
    }
}
