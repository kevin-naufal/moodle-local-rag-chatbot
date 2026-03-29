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
 * Draft persistence layer for publish flow.
 *
 * @package    local_chatbot
 * @copyright  2026
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class draft_repository {
    /** @var string */
    private const TABLE = 'local_chatbot_drafts';

    /**
     * Get draft by id.
     *
     * @param int $id
     * @return \stdClass
     */
    public function get_by_id(int $id): \stdClass {
        global $DB;
        return $DB->get_record(self::TABLE, ['id' => $id], '*', MUST_EXIST);
    }

    /**
     * Create draft record.
     *
     * @param int $courseid
     * @param int $userid
     * @param string $title
     * @param string $assignmenttype
     * @param int $questioncount
     * @param array $payload
     * @param string $status
     * @return int
     */
    public function create(
        int $courseid,
        int $userid,
        string $title,
        string $assignmenttype,
        int $questioncount,
        array $payload,
        string $status = 'draft'
    ): int {
        global $DB;
        $now = time();
        $record = (object)[
            'courseid' => $courseid,
            'userid' => $userid,
            'title' => trim($title),
            'assignment_type' => trim($assignmenttype),
            'question_count' => max(0, $questioncount),
            'draft_json' => json_encode($payload, JSON_UNESCAPED_UNICODE),
            'status' => trim($status),
            'published_cmid' => null,
            'error_message' => null,
            'published_at' => null,
            'timecreated' => $now,
            'timemodified' => $now,
        ];
        return (int)$DB->insert_record(self::TABLE, $record);
    }

    /**
     * Update status to published and store generated cmid.
     *
     * @param int $id
     * @param int $cmid
     * @return void
     */
    public function mark_published(int $id, int $cmid): void {
        global $DB;
        $now = time();
        $record = (object)[
            'id' => $id,
            'status' => 'published',
            'published_cmid' => $cmid,
            'published_at' => $now,
            'error_message' => null,
            'timemodified' => $now,
        ];
        $DB->update_record(self::TABLE, $record);
    }

    /**
     * Update status to failed and store short error message.
     *
     * @param int $id
     * @param string $message
     * @return void
     */
    public function mark_failed(int $id, string $message): void {
        global $DB;
        if (!$DB->record_exists(self::TABLE, ['id' => $id])) {
            return;
        }

        $shortmessage = \core_text::substr($message, 0, 4000);
        $record = (object)[
            'id' => $id,
            'status' => 'failed',
            'error_message' => $shortmessage,
            'timemodified' => time(),
        ];
        $DB->update_record(self::TABLE, $record);
    }
}
