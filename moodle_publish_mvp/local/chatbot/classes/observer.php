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

use local_chatbot\service\learning_profile_service;
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

        try {
            learning_profile_service::ingest_quiz_attempt($attemptid, $cmid, $courseid);
        } catch (\Throwable $e) {
            // Learning analytics should never break quiz submission flow.
            debugging('local_chatbot learning profile ingestion failed: ' . $e->getMessage(), DEBUG_DEVELOPER);
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
