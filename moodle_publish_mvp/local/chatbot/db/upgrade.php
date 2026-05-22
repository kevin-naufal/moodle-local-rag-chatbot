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

    // 2026050900: Remove obsolete task-generation and mastery storage.
    if ($oldversion < 2026050900) {
        $droptables = [
            'local_chatbot_drafts',
            'local_chatbot_std_profile',
            'local_chatbot_learn_events',
            'local_chatbot_weekly_snap',
            'local_chatbot_weight_map',
            'local_chatbot_weight_rule',
            'local_chatbot_weight_scheme',
        ];
        foreach ($droptables as $tablename) {
            $table = new xmldb_table($tablename);
            if ($dbman->table_exists($table)) {
                $dbman->drop_table($table);
            }
        }

        $likeparams = $DB->sql_like('name', ':masteryprefix', false);
        $DB->delete_records_select('config_plugins', "plugin = :plugin AND {$likeparams}", [
            'plugin' => 'local_chatbot',
            'masteryprefix' => 'mastery_policy_%',
        ]);

        upgrade_plugin_savepoint(true, 2026050900, 'local', 'chatbot');
    }

    // 2026052000: Add per-answer evaluation feedback table.
    if ($oldversion < 2026052000) {
        $feedbacktable = new xmldb_table('local_chatbot_eval_feedback');
        if (!$dbman->table_exists($feedbacktable)) {
            $feedbacktable->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
            $feedbacktable->add_field('userid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $feedbacktable->add_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $feedbacktable->add_field('request_id', XMLDB_TYPE_CHAR, '100', null, XMLDB_NOTNULL, null, '');
            $feedbacktable->add_field('chat_mode', XMLDB_TYPE_CHAR, '50', null, XMLDB_NOTNULL, null, '');
            $feedbacktable->add_field('question_id', XMLDB_TYPE_CHAR, '255', null, null, null, null);
            $feedbacktable->add_field('run_id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $feedbacktable->add_field('topic', XMLDB_TYPE_CHAR, '255', null, null, null, null);
            $feedbacktable->add_field('question_text', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null);
            $feedbacktable->add_field('answer_text', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null);
            $feedbacktable->add_field('sources_json', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null);
            $feedbacktable->add_field('correctness', XMLDB_TYPE_INTEGER, '2', null, XMLDB_NOTNULL, null, '0');
            $feedbacktable->add_field('groundedness', XMLDB_TYPE_INTEGER, '2', null, XMLDB_NOTNULL, null, '0');
            $feedbacktable->add_field('relevance', XMLDB_TYPE_INTEGER, '2', null, XMLDB_NOTNULL, null, '0');
            $feedbacktable->add_field('instruction_compliance', XMLDB_TYPE_INTEGER, '2', null, XMLDB_NOTNULL, null, '0');
            $feedbacktable->add_field('need_alignment', XMLDB_TYPE_INTEGER, '2', null, XMLDB_NOTNULL, null, '0');
            $feedbacktable->add_field('scaffolding_quality', XMLDB_TYPE_INTEGER, '2', null, XMLDB_NOTNULL, null, '0');
            $feedbacktable->add_field('clarity', XMLDB_TYPE_INTEGER, '2', null, XMLDB_NOTNULL, null, '0');
            $feedbacktable->add_field('comment_text', XMLDB_TYPE_TEXT, null, null, null, null, null);
            $feedbacktable->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $feedbacktable->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');

            $feedbacktable->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
            $feedbacktable->add_key('userid_fk', XMLDB_KEY_FOREIGN, ['userid'], 'user', ['id']);
            $feedbacktable->add_key('courseid_fk', XMLDB_KEY_FOREIGN, ['courseid'], 'course', ['id']);

            $feedbacktable->add_index('user_request_uix', XMLDB_INDEX_UNIQUE, ['userid', 'request_id']);
            $feedbacktable->add_index('course_time_idx', XMLDB_INDEX_NOTUNIQUE, ['courseid', 'timecreated']);
            $feedbacktable->add_index('question_mode_idx', XMLDB_INDEX_NOTUNIQUE, ['question_id', 'chat_mode']);

            $dbman->create_table($feedbacktable);
        }

        upgrade_plugin_savepoint(true, 2026052000, 'local', 'chatbot');
    }

    // 2026052200: Add automatic online per-query system performance snapshots.
    if ($oldversion < 2026052200) {
        $onlinetable = new xmldb_table('local_chatbot_online_eval');
        if (!$dbman->table_exists($onlinetable)) {
            $onlinetable->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
            $onlinetable->add_field('userid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $onlinetable->add_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $onlinetable->add_field('request_id', XMLDB_TYPE_CHAR, '100', null, XMLDB_NOTNULL, null, '');
            $onlinetable->add_field('chat_mode', XMLDB_TYPE_CHAR, '50', null, XMLDB_NOTNULL, null, '');
            $onlinetable->add_field('question_id', XMLDB_TYPE_CHAR, '255', null, null, null, null);
            $onlinetable->add_field('run_id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $onlinetable->add_field('topic', XMLDB_TYPE_CHAR, '255', null, null, null, null);
            $onlinetable->add_field('question_text', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null);
            $onlinetable->add_field('answer_text', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null);
            $onlinetable->add_field('sources_json', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null);
            $onlinetable->add_field('status', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'success');
            $onlinetable->add_field('predicted_behavior', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'unknown');
            $onlinetable->add_field('model_name', XMLDB_TYPE_CHAR, '255', null, null, null, null);
            $onlinetable->add_field('embedding_backend', XMLDB_TYPE_CHAR, '20', null, null, null, null);
            $onlinetable->add_field('embedding_model_name', XMLDB_TYPE_CHAR, '255', null, null, null, null);
            $onlinetable->add_field('latency_total', XMLDB_TYPE_NUMBER, '10, 3', null, XMLDB_NOTNULL, null, '0');
            $onlinetable->add_field('latency_retrieval', XMLDB_TYPE_NUMBER, '10, 3', null, XMLDB_NOTNULL, null, '0');
            $onlinetable->add_field('latency_generation', XMLDB_TYPE_NUMBER, '10, 3', null, XMLDB_NOTNULL, null, '0');
            $onlinetable->add_field('retrieved_context_count', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $onlinetable->add_field('source_count', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $onlinetable->add_field('answer_chars', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $onlinetable->add_field('error_message', XMLDB_TYPE_TEXT, null, null, null, null, null);
            $onlinetable->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
            $onlinetable->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');

            $onlinetable->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
            $onlinetable->add_key('userid_fk', XMLDB_KEY_FOREIGN, ['userid'], 'user', ['id']);
            $onlinetable->add_key('courseid_fk', XMLDB_KEY_FOREIGN, ['courseid'], 'course', ['id']);

            $onlinetable->add_index('user_request_uix', XMLDB_INDEX_UNIQUE, ['userid', 'request_id']);
            $onlinetable->add_index('course_time_idx', XMLDB_INDEX_NOTUNIQUE, ['courseid', 'timecreated']);
            $onlinetable->add_index('mode_time_idx', XMLDB_INDEX_NOTUNIQUE, ['chat_mode', 'timecreated']);
            $onlinetable->add_index('question_mode_idx', XMLDB_INDEX_NOTUNIQUE, ['question_id', 'chat_mode']);

            $dbman->create_table($onlinetable);
        }

        upgrade_plugin_savepoint(true, 2026052200, 'local', 'chatbot');
    }

    return true;
}
