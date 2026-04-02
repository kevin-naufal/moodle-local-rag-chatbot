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

defined('MOODLE_INTERNAL') || die();

/**
 * Upgrade script for local_chatbot.
 *
 * @param int $oldversion
 * @return bool
 */
function xmldb_local_chatbot_upgrade(int $oldversion): bool {
    global $DB;

    $dbman = $DB->get_manager();

    // 2026032900: Add drafts table for Generate -> Review -> Publish workflow.
    if ($oldversion < 2026032900) {
        $table = new xmldb_table('local_chatbot_drafts');

        if (!$dbman->table_exists($table)) {
            $table->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
            $table->add_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $table->add_field('userid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $table->add_field('title', XMLDB_TYPE_CHAR, '255', null, XMLDB_NOTNULL, null, '');
            $table->add_field('assignment_type', XMLDB_TYPE_CHAR, '50', null, XMLDB_NOTNULL, null, 'multiple_choice');
            $table->add_field('question_count', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $table->add_field('draft_json', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null);
            $table->add_field('status', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'draft');
            $table->add_field('published_cmid', XMLDB_TYPE_INTEGER, '10', null, null, null, null);
            $table->add_field('error_message', XMLDB_TYPE_TEXT, null, null, null, null, null);
            $table->add_field('published_at', XMLDB_TYPE_INTEGER, '10', null, null, null, null);
            $table->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $table->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');

            $table->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
            $table->add_key('courseid_fk', XMLDB_KEY_FOREIGN, ['courseid'], 'course', ['id']);
            $table->add_key('userid_fk', XMLDB_KEY_FOREIGN, ['userid'], 'user', ['id']);

            $table->add_index('status_idx', XMLDB_INDEX_NOTUNIQUE, ['status']);
            $table->add_index('course_status_idx', XMLDB_INDEX_NOTUNIQUE, ['courseid', 'status']);

            $dbman->create_table($table);
        } else {
            // Backfill missing fields/indexes safely for partially upgraded environments.
            $fields = [
                new xmldb_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('userid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('title', XMLDB_TYPE_CHAR, '255', null, XMLDB_NOTNULL, null, ''),
                new xmldb_field('assignment_type', XMLDB_TYPE_CHAR, '50', null, XMLDB_NOTNULL, null, 'multiple_choice'),
                new xmldb_field('question_count', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('draft_json', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null),
                new xmldb_field('status', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'draft'),
                new xmldb_field('published_cmid', XMLDB_TYPE_INTEGER, '10', null, null, null, null),
                new xmldb_field('error_message', XMLDB_TYPE_TEXT, null, null, null, null, null),
                new xmldb_field('published_at', XMLDB_TYPE_INTEGER, '10', null, null, null, null),
                new xmldb_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
            ];
            foreach ($fields as $field) {
                if (!$dbman->field_exists($table, $field)) {
                    $dbman->add_field($table, $field);
                }
            }

            $statusindex = new xmldb_index('status_idx', XMLDB_INDEX_NOTUNIQUE, ['status']);
            if (!$dbman->index_exists($table, $statusindex)) {
                $dbman->add_index($table, $statusindex);
            }
            $coursestatusindex = new xmldb_index('course_status_idx', XMLDB_INDEX_NOTUNIQUE, ['courseid', 'status']);
            if (!$dbman->index_exists($table, $coursestatusindex)) {
                $dbman->add_index($table, $coursestatusindex);
            }
        }

        upgrade_plugin_savepoint(true, 2026032900, 'local', 'chatbot');
    }

    // 2026040100: Add learning profile and learning events tables.
    if ($oldversion < 2026040100) {
        $profiletable = new xmldb_table('local_chatbot_std_profile');
        if (!$dbman->table_exists($profiletable)) {
            $profiletable->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
            $profiletable->add_field('userid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $profiletable->add_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $profiletable->add_field('topic', XMLDB_TYPE_CHAR, '255', null, XMLDB_NOTNULL, null, '');
            $profiletable->add_field('mastery', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0');
            $profiletable->add_field('accuracy_avg', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0');
            $profiletable->add_field('attempt_count', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $profiletable->add_field('avg_duration_seconds', XMLDB_TYPE_NUMBER, '10, 2', null, XMLDB_NOTNULL, null, '0');
            $profiletable->add_field('trend', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0');
            $profiletable->add_field('last_score', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0');
            $profiletable->add_field('last_event_time', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $profiletable->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $profiletable->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');

            $profiletable->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
            $profiletable->add_key('userid_fk', XMLDB_KEY_FOREIGN, ['userid'], 'user', ['id']);
            $profiletable->add_key('courseid_fk', XMLDB_KEY_FOREIGN, ['courseid'], 'course', ['id']);

            $profiletable->add_index('user_course_topic_uix', XMLDB_INDEX_UNIQUE, ['userid', 'courseid', 'topic']);
            $profiletable->add_index('course_topic_idx', XMLDB_INDEX_NOTUNIQUE, ['courseid', 'topic']);

            $dbman->create_table($profiletable);
        } else {
            $fields = [
                new xmldb_field('userid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('topic', XMLDB_TYPE_CHAR, '255', null, XMLDB_NOTNULL, null, ''),
                new xmldb_field('mastery', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('accuracy_avg', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('attempt_count', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('avg_duration_seconds', XMLDB_TYPE_NUMBER, '10, 2', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('trend', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('last_score', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('last_event_time', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
            ];
            foreach ($fields as $field) {
                if (!$dbman->field_exists($profiletable, $field)) {
                    $dbman->add_field($profiletable, $field);
                }
            }

            $usercoursetopicindex = new xmldb_index('user_course_topic_uix', XMLDB_INDEX_UNIQUE, ['userid', 'courseid', 'topic']);
            if (!$dbman->index_exists($profiletable, $usercoursetopicindex)) {
                $dbman->add_index($profiletable, $usercoursetopicindex);
            }
            $coursetopicindex = new xmldb_index('course_topic_idx', XMLDB_INDEX_NOTUNIQUE, ['courseid', 'topic']);
            if (!$dbman->index_exists($profiletable, $coursetopicindex)) {
                $dbman->add_index($profiletable, $coursetopicindex);
            }
        }

        $eventtable = new xmldb_table('local_chatbot_learn_events');
        if (!$dbman->table_exists($eventtable)) {
            $eventtable->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
            $eventtable->add_field('userid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $eventtable->add_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $eventtable->add_field('topic', XMLDB_TYPE_CHAR, '255', null, XMLDB_NOTNULL, null, '');
            $eventtable->add_field('module', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'quiz');
            $eventtable->add_field('event_type', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'quiz');
            $eventtable->add_field('cmid', XMLDB_TYPE_INTEGER, '10', null, null, null, null);
            $eventtable->add_field('activityid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $eventtable->add_field('attemptid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $eventtable->add_field('attempt_number', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $eventtable->add_field('score_raw', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0');
            $eventtable->add_field('score_max', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0');
            $eventtable->add_field('score_topic', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0');
            $eventtable->add_field('duration_seconds', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $eventtable->add_field('started_at', XMLDB_TYPE_INTEGER, '10', null, null, null, null);
            $eventtable->add_field('submitted_at', XMLDB_TYPE_INTEGER, '10', null, null, null, null);
            $eventtable->add_field('details_json', XMLDB_TYPE_TEXT, null, null, null, null, null);
            $eventtable->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');

            $eventtable->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
            $eventtable->add_key('userid_fk', XMLDB_KEY_FOREIGN, ['userid'], 'user', ['id']);
            $eventtable->add_key('courseid_fk', XMLDB_KEY_FOREIGN, ['courseid'], 'course', ['id']);

            $eventtable->add_index('module_attempt_uix', XMLDB_INDEX_UNIQUE, ['module', 'attemptid']);
            $eventtable->add_index('user_course_topic_time_idx', XMLDB_INDEX_NOTUNIQUE, ['userid', 'courseid', 'topic', 'timecreated']);
            $eventtable->add_index('event_type_time_idx', XMLDB_INDEX_NOTUNIQUE, ['event_type', 'timecreated']);

            $dbman->create_table($eventtable);
        } else {
            $fields = [
                new xmldb_field('userid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('topic', XMLDB_TYPE_CHAR, '255', null, XMLDB_NOTNULL, null, ''),
                new xmldb_field('module', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'quiz'),
                new xmldb_field('event_type', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'quiz'),
                new xmldb_field('cmid', XMLDB_TYPE_INTEGER, '10', null, null, null, null),
                new xmldb_field('activityid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('attemptid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('attempt_number', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('score_raw', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('score_max', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('score_topic', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('duration_seconds', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('started_at', XMLDB_TYPE_INTEGER, '10', null, null, null, null),
                new xmldb_field('submitted_at', XMLDB_TYPE_INTEGER, '10', null, null, null, null),
                new xmldb_field('details_json', XMLDB_TYPE_TEXT, null, null, null, null, null),
                new xmldb_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
            ];
            foreach ($fields as $field) {
                if (!$dbman->field_exists($eventtable, $field)) {
                    $dbman->add_field($eventtable, $field);
                }
            }

            $moduleattemptindex = new xmldb_index('module_attempt_uix', XMLDB_INDEX_UNIQUE, ['module', 'attemptid']);
            if (!$dbman->index_exists($eventtable, $moduleattemptindex)) {
                $dbman->add_index($eventtable, $moduleattemptindex);
            }
            $usercoursetopictimeindex = new xmldb_index('user_course_topic_time_idx', XMLDB_INDEX_NOTUNIQUE, ['userid', 'courseid', 'topic', 'timecreated']);
            if (!$dbman->index_exists($eventtable, $usercoursetopictimeindex)) {
                $dbman->add_index($eventtable, $usercoursetopictimeindex);
            }
            $eventtypetimeindex = new xmldb_index('event_type_time_idx', XMLDB_INDEX_NOTUNIQUE, ['event_type', 'timecreated']);
            if (!$dbman->index_exists($eventtable, $eventtypetimeindex)) {
                $dbman->add_index($eventtable, $eventtypetimeindex);
            }
        }

        upgrade_plugin_savepoint(true, 2026040100, 'local', 'chatbot');
    }

    // 2026040200: Add essay auto-grading result table.
    if ($oldversion < 2026040200) {
        $essaytable = new xmldb_table('local_chatbot_essay_grades');
        if (!$dbman->table_exists($essaytable)) {
            $essaytable->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
            $essaytable->add_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $essaytable->add_field('graderid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $essaytable->add_field('studentid', XMLDB_TYPE_INTEGER, '10', null, null, null, null);
            $essaytable->add_field('question_number', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '1');
            $essaytable->add_field('rubric_id', XMLDB_TYPE_CHAR, '100', null, XMLDB_NOTNULL, null, 'essay_default_v1');
            $essaytable->add_field('question_text', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null);
            $essaytable->add_field('expected_key_points', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null);
            $essaytable->add_field('student_answer', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null);
            $essaytable->add_field('grade_json', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null);
            $essaytable->add_field('overall_score', XMLDB_TYPE_NUMBER, '10, 2', null, XMLDB_NOTNULL, null, '0');
            $essaytable->add_field('confidence', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0');
            $essaytable->add_field('needs_manual_review', XMLDB_TYPE_INTEGER, '1', null, XMLDB_NOTNULL, null, '0');
            $essaytable->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');

            $essaytable->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
            $essaytable->add_key('courseid_fk', XMLDB_KEY_FOREIGN, ['courseid'], 'course', ['id']);
            $essaytable->add_key('graderid_fk', XMLDB_KEY_FOREIGN, ['graderid'], 'user', ['id']);
            $essaytable->add_key('studentid_fk', XMLDB_KEY_FOREIGN, ['studentid'], 'user', ['id']);

            $essaytable->add_index('course_time_idx', XMLDB_INDEX_NOTUNIQUE, ['courseid', 'timecreated']);
            $essaytable->add_index('student_time_idx', XMLDB_INDEX_NOTUNIQUE, ['studentid', 'timecreated']);
            $essaytable->add_index('manual_review_idx', XMLDB_INDEX_NOTUNIQUE, ['needs_manual_review', 'timecreated']);

            $dbman->create_table($essaytable);
        } else {
            $fields = [
                new xmldb_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('graderid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('studentid', XMLDB_TYPE_INTEGER, '10', null, null, null, null),
                new xmldb_field('question_number', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '1'),
                new xmldb_field('rubric_id', XMLDB_TYPE_CHAR, '100', null, XMLDB_NOTNULL, null, 'essay_default_v1'),
                new xmldb_field('question_text', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null),
                new xmldb_field('expected_key_points', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null),
                new xmldb_field('student_answer', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null),
                new xmldb_field('grade_json', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null),
                new xmldb_field('overall_score', XMLDB_TYPE_NUMBER, '10, 2', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('confidence', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('needs_manual_review', XMLDB_TYPE_INTEGER, '1', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
            ];
            foreach ($fields as $field) {
                if (!$dbman->field_exists($essaytable, $field)) {
                    $dbman->add_field($essaytable, $field);
                }
            }

            $coursetimeindex = new xmldb_index('course_time_idx', XMLDB_INDEX_NOTUNIQUE, ['courseid', 'timecreated']);
            if (!$dbman->index_exists($essaytable, $coursetimeindex)) {
                $dbman->add_index($essaytable, $coursetimeindex);
            }
            $studenttimeindex = new xmldb_index('student_time_idx', XMLDB_INDEX_NOTUNIQUE, ['studentid', 'timecreated']);
            if (!$dbman->index_exists($essaytable, $studenttimeindex)) {
                $dbman->add_index($essaytable, $studenttimeindex);
            }
            $manualreviewindex = new xmldb_index('manual_review_idx', XMLDB_INDEX_NOTUNIQUE, ['needs_manual_review', 'timecreated']);
            if (!$dbman->index_exists($essaytable, $manualreviewindex)) {
                $dbman->add_index($essaytable, $manualreviewindex);
            }
        }

        upgrade_plugin_savepoint(true, 2026040200, 'local', 'chatbot');
    }

    // 2026040201: Add essay auto-grade config table and submission context fields.
    if ($oldversion < 2026040201) {
        $essaytable = new xmldb_table('local_chatbot_essay_grades');
        if ($dbman->table_exists($essaytable)) {
            $fields = [
                new xmldb_field('assignmentid', XMLDB_TYPE_INTEGER, '10', null, null, null, null, 'rubric_id'),
                new xmldb_field('cmid', XMLDB_TYPE_INTEGER, '10', null, null, null, null, 'assignmentid'),
                new xmldb_field('submissionid', XMLDB_TYPE_INTEGER, '10', null, null, null, null, 'cmid'),
                new xmldb_field('attemptnumber', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0', 'submissionid'),
                new xmldb_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0', 'timecreated'),
            ];
            foreach ($fields as $field) {
                if (!$dbman->field_exists($essaytable, $field)) {
                    $dbman->add_field($essaytable, $field);
                }
            }

            $submissionquestionindex = new xmldb_index(
                'submission_question_idx',
                XMLDB_INDEX_NOTUNIQUE,
                ['submissionid', 'question_number']
            );
            if (!$dbman->index_exists($essaytable, $submissionquestionindex)) {
                $dbman->add_index($essaytable, $submissionquestionindex);
            }

            $assignstudentattemptindex = new xmldb_index(
                'assign_student_attempt_idx',
                XMLDB_INDEX_NOTUNIQUE,
                ['assignmentid', 'studentid', 'attemptnumber']
            );
            if (!$dbman->index_exists($essaytable, $assignstudentattemptindex)) {
                $dbman->add_index($essaytable, $assignstudentattemptindex);
            }
        }

        $cfgtable = new xmldb_table('local_chatbot_essay_autocfg');
        if (!$dbman->table_exists($cfgtable)) {
            $cfgtable->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
            $cfgtable->add_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $cfgtable->add_field('cmid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $cfgtable->add_field('assignid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $cfgtable->add_field('enabled', XMLDB_TYPE_INTEGER, '1', null, XMLDB_NOTNULL, null, '1');
            $cfgtable->add_field('rubric_id', XMLDB_TYPE_CHAR, '100', null, XMLDB_NOTNULL, null, 'essay_default_v1');
            $cfgtable->add_field('config_json', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null);
            $cfgtable->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $cfgtable->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');

            $cfgtable->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
            $cfgtable->add_key('courseid_fk', XMLDB_KEY_FOREIGN, ['courseid'], 'course', ['id']);

            $cfgtable->add_index('assignid_uix', XMLDB_INDEX_UNIQUE, ['assignid']);
            $cfgtable->add_index('cmid_idx', XMLDB_INDEX_NOTUNIQUE, ['cmid']);
            $cfgtable->add_index('course_enabled_idx', XMLDB_INDEX_NOTUNIQUE, ['courseid', 'enabled']);

            $dbman->create_table($cfgtable);
        } else {
            $fields = [
                new xmldb_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('cmid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('assignid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('enabled', XMLDB_TYPE_INTEGER, '1', null, XMLDB_NOTNULL, null, '1'),
                new xmldb_field('rubric_id', XMLDB_TYPE_CHAR, '100', null, XMLDB_NOTNULL, null, 'essay_default_v1'),
                new xmldb_field('config_json', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null),
                new xmldb_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
            ];
            foreach ($fields as $field) {
                if (!$dbman->field_exists($cfgtable, $field)) {
                    $dbman->add_field($cfgtable, $field);
                }
            }

            $assignidindex = new xmldb_index('assignid_uix', XMLDB_INDEX_UNIQUE, ['assignid']);
            if (!$dbman->index_exists($cfgtable, $assignidindex)) {
                $dbman->add_index($cfgtable, $assignidindex);
            }
            $cmidindex = new xmldb_index('cmid_idx', XMLDB_INDEX_NOTUNIQUE, ['cmid']);
            if (!$dbman->index_exists($cfgtable, $cmidindex)) {
                $dbman->add_index($cfgtable, $cmidindex);
            }
            $courseenabledindex = new xmldb_index('course_enabled_idx', XMLDB_INDEX_NOTUNIQUE, ['courseid', 'enabled']);
            if (!$dbman->index_exists($cfgtable, $courseenabledindex)) {
                $dbman->add_index($cfgtable, $courseenabledindex);
            }
        }

        upgrade_plugin_savepoint(true, 2026040201, 'local', 'chatbot');
    }

    return true;
}
