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

    // 2026040202: Add weekly mastery snapshot table.
    if ($oldversion < 2026040202) {
        $snaptable = new xmldb_table('local_chatbot_weekly_snap');
        if (!$dbman->table_exists($snaptable)) {
            $snaptable->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
            $snaptable->add_field('userid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $snaptable->add_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $snaptable->add_field('topic', XMLDB_TYPE_CHAR, '255', null, XMLDB_NOTNULL, null, '');
            $snaptable->add_field('week_start', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $snaptable->add_field('mastery', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0');
            $snaptable->add_field('accuracy_avg', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0');
            $snaptable->add_field('attempt_count', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $snaptable->add_field('first_event_time', XMLDB_TYPE_INTEGER, '10', null, null, null, null);
            $snaptable->add_field('last_event_time', XMLDB_TYPE_INTEGER, '10', null, null, null, null);
            $snaptable->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $snaptable->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');

            $snaptable->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
            $snaptable->add_key('userid_fk', XMLDB_KEY_FOREIGN, ['userid'], 'user', ['id']);
            $snaptable->add_key('courseid_fk', XMLDB_KEY_FOREIGN, ['courseid'], 'course', ['id']);

            $snaptable->add_index(
                'user_course_topic_week_uix',
                XMLDB_INDEX_UNIQUE,
                ['userid', 'courseid', 'topic', 'week_start']
            );
            $snaptable->add_index('course_week_idx', XMLDB_INDEX_NOTUNIQUE, ['courseid', 'week_start']);
            $snaptable->add_index('user_week_idx', XMLDB_INDEX_NOTUNIQUE, ['userid', 'week_start']);

            $dbman->create_table($snaptable);
        } else {
            $fields = [
                new xmldb_field('userid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('topic', XMLDB_TYPE_CHAR, '255', null, XMLDB_NOTNULL, null, ''),
                new xmldb_field('week_start', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('mastery', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('accuracy_avg', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('attempt_count', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('first_event_time', XMLDB_TYPE_INTEGER, '10', null, null, null, null),
                new xmldb_field('last_event_time', XMLDB_TYPE_INTEGER, '10', null, null, null, null),
                new xmldb_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
            ];
            foreach ($fields as $field) {
                if (!$dbman->field_exists($snaptable, $field)) {
                    $dbman->add_field($snaptable, $field);
                }
            }

            $uniqueindex = new xmldb_index(
                'user_course_topic_week_uix',
                XMLDB_INDEX_UNIQUE,
                ['userid', 'courseid', 'topic', 'week_start']
            );
            if (!$dbman->index_exists($snaptable, $uniqueindex)) {
                $dbman->add_index($snaptable, $uniqueindex);
            }
            $courseweekindex = new xmldb_index('course_week_idx', XMLDB_INDEX_NOTUNIQUE, ['courseid', 'week_start']);
            if (!$dbman->index_exists($snaptable, $courseweekindex)) {
                $dbman->add_index($snaptable, $courseweekindex);
            }
            $userweekindex = new xmldb_index('user_week_idx', XMLDB_INDEX_NOTUNIQUE, ['userid', 'week_start']);
            if (!$dbman->index_exists($snaptable, $userweekindex)) {
                $dbman->add_index($snaptable, $userweekindex);
            }
        }

        // Initial backfill: create one snapshot row for current week from existing profile rows.
        if ($dbman->table_exists(new xmldb_table('local_chatbot_std_profile'))) {
            $profiles = $DB->get_records(
                'local_chatbot_std_profile',
                null,
                '',
                'id,userid,courseid,topic,mastery,accuracy_avg,attempt_count,last_event_time,timecreated,timemodified'
            );
            $now = time();
            foreach ($profiles as $profile) {
                $eventtime = (int)$profile->last_event_time;
                if ($eventtime <= 0) {
                    $eventtime = (int)$profile->timemodified > 0 ? (int)$profile->timemodified : (int)$profile->timecreated;
                }
                if ($eventtime <= 0) {
                    $eventtime = $now;
                }

                $dt = new DateTime('@' . $eventtime);
                $dt->setTimezone(new DateTimeZone('UTC'));
                $day = (int)$dt->format('N');
                $dt->setTime(0, 0, 0);
                if ($day > 1) {
                    $dt->modify('-' . ($day - 1) . ' days');
                }
                $weekstart = (int)$dt->getTimestamp();

                $firstevent = (int)$DB->get_field_sql(
                    "SELECT MIN(submitted_at)
                       FROM {local_chatbot_learn_events}
                      WHERE userid = :userid
                        AND courseid = :courseid
                        AND topic = :topic",
                    [
                        'userid' => (int)$profile->userid,
                        'courseid' => (int)$profile->courseid,
                        'topic' => (string)$profile->topic,
                    ]
                );
                $lastevent = (int)$DB->get_field_sql(
                    "SELECT MAX(submitted_at)
                       FROM {local_chatbot_learn_events}
                      WHERE userid = :userid
                        AND courseid = :courseid
                        AND topic = :topic",
                    [
                        'userid' => (int)$profile->userid,
                        'courseid' => (int)$profile->courseid,
                        'topic' => (string)$profile->topic,
                    ]
                );

                if ($DB->record_exists(
                    'local_chatbot_weekly_snap',
                    [
                        'userid' => (int)$profile->userid,
                        'courseid' => (int)$profile->courseid,
                        'topic' => (string)$profile->topic,
                        'week_start' => $weekstart,
                    ]
                )) {
                    continue;
                }

                $record = (object)[
                    'userid' => (int)$profile->userid,
                    'courseid' => (int)$profile->courseid,
                    'topic' => (string)$profile->topic,
                    'week_start' => $weekstart,
                    'mastery' => (float)$profile->mastery,
                    'accuracy_avg' => (float)$profile->accuracy_avg,
                    'attempt_count' => (int)$profile->attempt_count,
                    'first_event_time' => $firstevent > 0 ? $firstevent : null,
                    'last_event_time' => $lastevent > 0 ? $lastevent : null,
                    'timecreated' => $now,
                    'timemodified' => $now,
                ];
                $DB->insert_record('local_chatbot_weekly_snap', $record);
            }
        }

        upgrade_plugin_savepoint(true, 2026040202, 'local', 'chatbot');
    }

    // 2026041900: Add weighting scheme tables for UI-only grading configuration.
    if ($oldversion < 2026041900) {
        $schemetable = new xmldb_table('local_chatbot_weight_scheme');
        if (!$dbman->table_exists($schemetable)) {
            $schemetable->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
            $schemetable->add_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $schemetable->add_field('name', XMLDB_TYPE_CHAR, '255', null, XMLDB_NOTNULL, null, '');
            $schemetable->add_field('is_active', XMLDB_TYPE_INTEGER, '1', null, XMLDB_NOTNULL, null, '1');
            $schemetable->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $schemetable->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');

            $schemetable->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
            $schemetable->add_key('courseid_fk', XMLDB_KEY_FOREIGN, ['courseid'], 'course', ['id']);
            $schemetable->add_index('course_active_idx', XMLDB_INDEX_NOTUNIQUE, ['courseid', 'is_active']);

            $dbman->create_table($schemetable);
        } else {
            $fields = [
                new xmldb_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('name', XMLDB_TYPE_CHAR, '255', null, XMLDB_NOTNULL, null, ''),
                new xmldb_field('is_active', XMLDB_TYPE_INTEGER, '1', null, XMLDB_NOTNULL, null, '1'),
                new xmldb_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
            ];
            foreach ($fields as $field) {
                if (!$dbman->field_exists($schemetable, $field)) {
                    $dbman->add_field($schemetable, $field);
                }
            }

            $courseactiveindex = new xmldb_index('course_active_idx', XMLDB_INDEX_NOTUNIQUE, ['courseid', 'is_active']);
            if (!$dbman->index_exists($schemetable, $courseactiveindex)) {
                $dbman->add_index($schemetable, $courseactiveindex);
            }
        }

        $ruletable = new xmldb_table('local_chatbot_weight_rule');
        if (!$dbman->table_exists($ruletable)) {
            $ruletable->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
            $ruletable->add_field('schemeid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $ruletable->add_field('level', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'category');
            $ruletable->add_field('category', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, '');
            $ruletable->add_field('subtype', XMLDB_TYPE_CHAR, '30', null, XMLDB_NOTNULL, null, 'all');
            $ruletable->add_field('source', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'all');
            $ruletable->add_field('weight_percent', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0');
            $ruletable->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $ruletable->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');

            $ruletable->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
            $ruletable->add_key('schemeid_fk', XMLDB_KEY_FOREIGN, ['schemeid'], 'local_chatbot_weight_scheme', ['id']);
            $ruletable->add_index(
                'scheme_level_bucket_uix',
                XMLDB_INDEX_UNIQUE,
                ['schemeid', 'level', 'category', 'subtype', 'source']
            );
            $ruletable->add_index('scheme_level_idx', XMLDB_INDEX_NOTUNIQUE, ['schemeid', 'level']);

            $dbman->create_table($ruletable);
        } else {
            $fields = [
                new xmldb_field('schemeid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('level', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'category'),
                new xmldb_field('category', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, ''),
                new xmldb_field('subtype', XMLDB_TYPE_CHAR, '30', null, XMLDB_NOTNULL, null, 'all'),
                new xmldb_field('source', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'all'),
                new xmldb_field('weight_percent', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
            ];
            foreach ($fields as $field) {
                if (!$dbman->field_exists($ruletable, $field)) {
                    $dbman->add_field($ruletable, $field);
                }
            }

            $uniqueruleindex = new xmldb_index(
                'scheme_level_bucket_uix',
                XMLDB_INDEX_UNIQUE,
                ['schemeid', 'level', 'category', 'subtype', 'source']
            );
            if (!$dbman->index_exists($ruletable, $uniqueruleindex)) {
                $dbman->add_index($ruletable, $uniqueruleindex);
            }
            $schemelevelindex = new xmldb_index('scheme_level_idx', XMLDB_INDEX_NOTUNIQUE, ['schemeid', 'level']);
            if (!$dbman->index_exists($ruletable, $schemelevelindex)) {
                $dbman->add_index($ruletable, $schemelevelindex);
            }
        }

        $maptable = new xmldb_table('local_chatbot_weight_map');
        if (!$dbman->table_exists($maptable)) {
            $maptable->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
            $maptable->add_field('schemeid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $maptable->add_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $maptable->add_field('cmid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $maptable->add_field('module', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, '');
            $maptable->add_field('activityname', XMLDB_TYPE_CHAR, '255', null, XMLDB_NOTNULL, null, '');
            $maptable->add_field('category', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, '');
            $maptable->add_field('subtype', XMLDB_TYPE_CHAR, '30', null, XMLDB_NOTNULL, null, '');
            $maptable->add_field('source', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, '');
            $maptable->add_field('item_weight_percent', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '100');
            $maptable->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $maptable->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');

            $maptable->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
            $maptable->add_key('schemeid_fk', XMLDB_KEY_FOREIGN, ['schemeid'], 'local_chatbot_weight_scheme', ['id']);
            $maptable->add_key('courseid_fk', XMLDB_KEY_FOREIGN, ['courseid'], 'course', ['id']);
            $maptable->add_index('scheme_cmid_uix', XMLDB_INDEX_UNIQUE, ['schemeid', 'cmid']);
            $maptable->add_index('course_bucket_idx', XMLDB_INDEX_NOTUNIQUE, ['courseid', 'category', 'subtype', 'source']);

            $dbman->create_table($maptable);
        } else {
            $fields = [
                new xmldb_field('schemeid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('cmid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('module', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, ''),
                new xmldb_field('activityname', XMLDB_TYPE_CHAR, '255', null, XMLDB_NOTNULL, null, ''),
                new xmldb_field('category', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, ''),
                new xmldb_field('subtype', XMLDB_TYPE_CHAR, '30', null, XMLDB_NOTNULL, null, ''),
                new xmldb_field('source', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, ''),
                new xmldb_field('item_weight_percent', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '100'),
                new xmldb_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
                new xmldb_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0'),
            ];
            foreach ($fields as $field) {
                if (!$dbman->field_exists($maptable, $field)) {
                    $dbman->add_field($maptable, $field);
                }
            }

            $schemecmidindex = new xmldb_index('scheme_cmid_uix', XMLDB_INDEX_UNIQUE, ['schemeid', 'cmid']);
            if (!$dbman->index_exists($maptable, $schemecmidindex)) {
                $dbman->add_index($maptable, $schemecmidindex);
            }
            $coursebucketindex = new xmldb_index(
                'course_bucket_idx',
                XMLDB_INDEX_NOTUNIQUE,
                ['courseid', 'category', 'subtype', 'source']
            );
            if (!$dbman->index_exists($maptable, $coursebucketindex)) {
                $dbman->add_index($maptable, $coursebucketindex);
            }
        }

        upgrade_plugin_savepoint(true, 2026041900, 'local', 'chatbot');
    }

    // 2026041901: Refresh plugin metadata/event cache for auto default activity weighting.
    if ($oldversion < 2026041901) {
        upgrade_plugin_savepoint(true, 2026041901, 'local', 'chatbot');
    }

    return true;
}
