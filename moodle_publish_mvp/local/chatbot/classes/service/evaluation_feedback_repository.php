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
 * Persistence for per-answer evaluation feedback.
 *
 * @package    local_chatbot
 * @copyright  2026
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class evaluation_feedback_repository {
    /** @var string */
    private const TABLE = 'local_chatbot_eval_feedback';

    /**
     * Create or update one evaluation-feedback record.
     *
     * @param array $payload
     * @return int
     */
    public function upsert(array $payload): int {
        global $DB;

        $now = time();
        $userid = max(0, (int)($payload['userid'] ?? 0));
        $requestid = trim((string)($payload['request_id'] ?? ''));
        $record = (object)[
            'userid' => $userid,
            'courseid' => max(0, (int)($payload['courseid'] ?? 0)),
            'request_id' => $requestid,
            'chat_mode' => trim((string)($payload['chat_mode'] ?? '')),
            'question_id' => trim((string)($payload['question_id'] ?? '')),
            'run_id' => max(0, (int)($payload['run_id'] ?? 0)),
            'topic' => trim((string)($payload['topic'] ?? '')),
            'question_text' => trim((string)($payload['question_text'] ?? '')),
            'answer_text' => trim((string)($payload['answer_text'] ?? '')),
            'sources_json' => json_encode($payload['sources'] ?? [], JSON_UNESCAPED_UNICODE),
            'correctness' => max(1, min(5, (int)($payload['correctness'] ?? 1))),
            'groundedness' => max(1, min(5, (int)($payload['groundedness'] ?? 1))),
            'relevance' => max(1, min(5, (int)($payload['relevance'] ?? 1))),
            'instruction_compliance' => max(1, min(5, (int)($payload['instruction_compliance'] ?? 1))),
            'need_alignment' => max(1, min(5, (int)($payload['need_alignment'] ?? 1))),
            'scaffolding_quality' => max(1, min(5, (int)($payload['scaffolding_quality'] ?? 1))),
            'clarity' => max(1, min(5, (int)($payload['clarity'] ?? 1))),
            'comment_text' => trim((string)($payload['comment_text'] ?? '')),
            'timemodified' => $now,
        ];

        $existing = null;
        if ($userid > 0 && $requestid !== '') {
            $existing = $DB->get_record(
                self::TABLE,
                ['userid' => $userid, 'request_id' => $requestid],
                'id',
                IGNORE_MISSING
            );
        }

        if ($existing) {
            $record->id = (int)$existing->id;
            $DB->update_record(self::TABLE, $record);
            return (int)$record->id;
        }

        $record->timecreated = $now;
        return (int)$DB->insert_record(self::TABLE, $record);
    }
}
