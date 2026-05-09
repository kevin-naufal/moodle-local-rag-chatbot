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

namespace local_chatbot;

defined('MOODLE_INTERNAL') || die();

use local_chatbot\service\essay_submission_autograde_service;
use local_chatbot\service\essay_quiz_autograde_service;

/**
 * Event observers for local_chatbot.
 *
 * @package    local_chatbot
 * @copyright  2026
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class observer {
    /**
     * Queue one pre-parse task for a course.
     *
     * @param int $courseid
     * @param int $userid
     * @param string $reason
     * @return void
     */
    private static function queue_preparse_course_task(int $courseid, int $userid, string $reason): void {
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

        $task = new \local_chatbot\task\preparse_course_task();
        $task->set_component('local_chatbot');
        $task->set_custom_data([
            'courseid' => $courseid,
            'userid' => $userid,
            'reason' => trim($reason),
        ]);
        \core\task\manager::queue_adhoc_task($task, true);
    }

    /**
     * Determine whether a resource module contains at least one PDF/TXT file.
     *
     * @param \cm_info|\stdClass $cm
     * @return bool
     */
    private static function resource_has_parseable_file($cm): bool {
        if (!$cm || empty($cm->id)) {
            return false;
        }
        $cmcontext = \context_module::instance((int)$cm->id, IGNORE_MISSING);
        if (!$cmcontext) {
            return false;
        }
        $fs = get_file_storage();
        $files = $fs->get_area_files(
            $cmcontext->id,
            'mod_resource',
            'content',
            0,
            'filename ASC',
            false
        );
        if (!$files) {
            return false;
        }
        foreach ($files as $file) {
            $filename = (string)$file->get_filename();
            $ext = \core_text::strtolower((string)pathinfo($filename, PATHINFO_EXTENSION));
            if ($ext === 'pdf' || $ext === 'txt') {
                return true;
            }
        }
        return false;
    }

    /**
     * Queue pre-parse when resource module contains parseable file.
     *
     * @param int $cmid
     * @param int $courseid
     * @param int $userid
     * @param string $reason
     * @return void
     */
    private static function queue_preparse_if_resource_module(
        int $cmid,
        int $courseid,
        int $userid,
        string $reason
    ): void {
        global $DB;

        if ($cmid <= 0 || $courseid <= 0) {
            return;
        }

        $cm = get_coursemodule_from_id('', $cmid, $courseid, false, IGNORE_MISSING);
        if (!$cm) {
            return;
        }

        $modname = '';
        if (!empty($cm->modname)) {
            $modname = (string)$cm->modname;
        } else if (!empty($cm->module)) {
            $modname = (string)$DB->get_field('modules', 'name', ['id' => (int)$cm->module], IGNORE_MISSING);
        }
        if ($modname !== 'resource') {
            return;
        }
        if (!self::resource_has_parseable_file($cm)) {
            return;
        }

        self::queue_preparse_course_task($courseid, $userid, $reason);
    }

    /**
     * Handle new course module creation for auto pre-parse refresh.
     *
     * @param \core\event\course_module_created $event
     * @return void
     */
    public static function course_module_created(\core\event\course_module_created $event): void {
        $cmid = (int)$event->objectid;
        $courseid = (int)$event->courseid;
        if ($cmid <= 0 || $courseid <= 0) {
            return;
        }

        try {
            self::queue_preparse_if_resource_module($cmid, $courseid, (int)$event->userid, 'course_module_created');
        } catch (\Throwable $e) {
            debugging('local_chatbot preparse on module create failed: ' . $e->getMessage(), DEBUG_DEVELOPER);
        }
    }

    /**
     * Handle course module updated event for auto pre-parse refresh.
     *
     * @param \core\event\course_module_updated $event
     * @return void
     */
    public static function course_module_updated(\core\event\course_module_updated $event): void {
        $cmid = (int)$event->objectid;
        $courseid = (int)$event->courseid;
        if ($cmid <= 0 || $courseid <= 0) {
            return;
        }

        try {
            self::queue_preparse_if_resource_module($cmid, $courseid, (int)$event->userid, 'course_module_updated');
        } catch (\Throwable $e) {
            debugging('local_chatbot preparse on module update failed: ' . $e->getMessage(), DEBUG_DEVELOPER);
        }
    }

    /**
     * Handle user login event and queue pre-parse for accessible courses.
     *
     * @param \core\event\user_loggedin $event
     * @return void
     */
    public static function user_loggedin(\core\event\user_loggedin $event): void {
        global $CFG;

        $userid = (int)$event->userid;
        if ($userid <= 0) {
            return;
        }

        require_once($CFG->dirroot . '/enrol/locallib.php');
        require_once($CFG->dirroot . '/local/chatbot/locallib.php');

        $courses = enrol_get_users_courses($userid, true, 'id');
        if (!$courses) {
            return;
        }

        foreach ($courses as $course) {
            $courseid = (int)$course->id;
            if ($courseid <= 1) {
                continue;
            }
            if (!local_chatbot_user_can_access_course_materials($courseid, $userid)) {
                continue;
            }
            $pdfs = local_chatbot_list_course_pdfs($courseid, $userid);
            if (empty($pdfs)) {
                continue;
            }
            self::queue_preparse_course_task($courseid, $userid, 'user_loggedin');
        }
    }

    /**
     * Handle quiz attempt submitted event.
     *
     * @param \mod_quiz\event\attempt_submitted $event
     * @return void
     */
    public static function quiz_attempt_submitted(\mod_quiz\event\attempt_submitted $event): void {
        $attemptid = (int)$event->objectid;
        $cmid = (int)$event->contextinstanceid;
        $courseid = (int)$event->courseid;
        $actorid = (int)$event->userid;

        if ($attemptid <= 0 || $courseid <= 0) {
            return;
        }

        try {
            essay_quiz_autograde_service::ingest_attempt($attemptid, $cmid, $courseid, $actorid);
        } catch (\Throwable $e) {
            // Essay auto-grading should never break quiz submission flow.
            debugging('local_chatbot quiz essay auto-grade failed: ' . $e->getMessage(), DEBUG_DEVELOPER);
        }
    }

    /**
     * Handle assignment submission event and trigger essay auto-grading when enabled.
     *
     * @param \mod_assign\event\assessable_submitted $event
     * @return void
     */
    public static function assign_assessable_submitted(\mod_assign\event\assessable_submitted $event): void {
        $submissionid = (int)$event->objectid;
        $cmid = (int)$event->contextinstanceid;
        $courseid = (int)$event->courseid;
        $actorid = (int)$event->userid;

        if ($submissionid <= 0 || $cmid <= 0 || $courseid <= 0) {
            return;
        }

        try {
            essay_submission_autograde_service::ingest_submission($submissionid, $cmid, $courseid, $actorid);
        } catch (\Throwable $e) {
            // Auto-grading should not block submission flow.
            debugging('local_chatbot essay auto-grade ingestion failed: ' . $e->getMessage(), DEBUG_DEVELOPER);
        }
    }

}
