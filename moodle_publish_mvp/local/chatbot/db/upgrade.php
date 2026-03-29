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

    return true;
}
