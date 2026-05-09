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

namespace local_chatbot\task;

defined('MOODLE_INTERNAL') || die();

/**
 * Adhoc task: sync course materials into chatbot data dir and warm RAG index cache.
 *
 * @package    local_chatbot
 * @copyright  2026
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class preparse_course_task extends \core\task\adhoc_task {
    /**
     * Task display name.
     *
     * @return string
     */
    public function get_name(): string {
        return 'Local Chatbot course pre-parse';
    }

    /**
     * Execute task.
     *
     * @return void
     */
    public function execute(): void {
        global $CFG;

        require_once($CFG->dirroot . '/local/chatbot/locallib.php');

        $data = (object)$this->get_custom_data();
        $courseid = (int)($data->courseid ?? 0);
        $userid = (int)($data->userid ?? 0);
        $topic = trim((string)($data->topic ?? ''));

        if ($courseid <= 0) {
            return;
        }

        if ($userid <= 0) {
            $admin = get_admin();
            $userid = $admin ? (int)$admin->id : 0;
        }
        if ($userid <= 0) {
            return;
        }

        try {
            local_chatbot_preparse_course_materials($courseid, $userid, $topic);
        } catch (\Throwable $e) {
            debugging('local_chatbot preparse task failed: ' . $e->getMessage(), DEBUG_DEVELOPER);
        }
    }
}

