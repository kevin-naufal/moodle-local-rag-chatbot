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
 * Parses markdown/text assignment draft into structured payload.
 *
 * @package    local_chatbot
 * @copyright  2026
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class markdown_draft_parser {
    /**
     * Parse draft text into draft_json payload.
     *
     * @param string $rawtext
     * @return array
     */
    public function parse(string $rawtext): array {
        $sections = $this->split_into_sections($rawtext);
        $questions = $this->parse_questions($sections['question_list'] ?? '');
        $answerkey = $this->parse_answer_key($sections['answer_key'] ?? '');

        return [
            'assignment_title' => $this->parse_title($sections['assignment_title'] ?? ''),
            'learning_objectives' => $this->parse_plain_list($sections['learning_objectives'] ?? ''),
            'instructions' => $this->collapse_lines($sections['instructions'] ?? ''),
            'questions' => $questions,
            'answer_key' => $answerkey,
            'grading_rubric' => $this->parse_plain_list($sections['grading_rubric'] ?? ''),
        ];
    }

    /**
     * Split text by known draft headers.
     *
     * @param string $rawtext
     * @return array
     */
    private function split_into_sections(string $rawtext): array {
        $normalized = str_replace("\r\n", "\n", $rawtext);
        $lines = explode("\n", $normalized);
        $sections = [];
        $current = null;

        foreach ($lines as $line) {
            $header = $this->detect_header($line);
            if ($header !== null) {
                $current = $header;
                if (!array_key_exists($current, $sections)) {
                    $sections[$current] = [];
                }
                $inlinecontent = $this->extract_inline_header_content($line);
                if ($inlinecontent !== '') {
                    $sections[$current][] = $inlinecontent;
                }
                continue;
            }
            if ($current !== null) {
                $sections[$current][] = $line;
            }
        }

        $result = [];
        foreach ($sections as $name => $contentlines) {
            $result[$name] = trim(implode("\n", $contentlines));
        }
        return $result;
    }

    /**
     * Detect canonical section header from a line.
     *
     * @param string $line
     * @return string|null
     */
    private function detect_header(string $line): ?string {
        $trimmed = trim($line);
        if ($trimmed === '') {
            return null;
        }

        $normalized = strtolower(trim($this->remove_markdown_emphasis($trimmed)));
        $normalized = preg_replace('/\s+/', ' ', (string)$normalized);

        $map = [
            'assignment title' => 'assignment_title',
            'judul tugas' => 'assignment_title',
            'learning objectives' => 'learning_objectives',
            'tujuan pembelajaran' => 'learning_objectives',
            'instructions for students' => 'instructions',
            'instruksi untuk siswa' => 'instructions',
            'question list' => 'question_list',
            'daftar soal' => 'question_list',
            'answer key' => 'answer_key',
            'kunci jawaban' => 'answer_key',
            'grading rubric' => 'grading_rubric',
            'rubrik penilaian' => 'grading_rubric',
        ];

        foreach ($map as $label => $canonical) {
            if ($normalized === $label || str_starts_with($normalized, $label . ':')) {
                return $canonical;
            }
        }

        return null;
    }

    /**
     * Extract same-line content after section header, e.g. "Assignment Title: My Title".
     *
     * @param string $line
     * @return string
     */
    private function extract_inline_header_content(string $line): string {
        $clean = trim($this->remove_markdown_emphasis($line));
        $parts = explode(':', $clean, 2);
        if (count($parts) < 2) {
            return '';
        }
        return trim((string)$parts[1]);
    }

    /**
     * Parse title line.
     *
     * @param string $raw
     * @return string
     */
    private function parse_title(string $raw): string {
        $line = trim($this->first_non_empty_line($raw));
        return $this->remove_markdown_emphasis($line);
    }

    /**
     * Parse list-like section into plain array entries.
     *
     * @param string $raw
     * @return array
     */
    private function parse_plain_list(string $raw): array {
        $items = [];
        foreach (explode("\n", $raw) as $line) {
            $trim = trim($line);
            if ($trim === '') {
                continue;
            }
            $trim = preg_replace('/^\s*(?:[-*]\s+|\d+[.)]\s+)/', '', $trim);
            $trim = trim($this->remove_markdown_emphasis((string)$trim));
            if ($trim !== '') {
                $items[] = $trim;
            }
        }
        return $items;
    }

    /**
     * Parse multiple-choice questions with A-D options.
     *
     * @param string $raw
     * @return array
     */
    private function parse_questions(string $raw): array {
        $questions = [];
        $current = null;
        $lines = explode("\n", $raw);

        foreach ($lines as $line) {
            $trim = trim($line);
            if ($trim === '') {
                continue;
            }

            if (preg_match('/^\s*(\d+)[.)]\s*(.+)$/', $trim, $matches)) {
                if ($current !== null) {
                    $questions[] = $current;
                }
                $current = [
                    'number' => (int)$matches[1],
                    'stem' => $this->remove_markdown_emphasis(trim($matches[2])),
                    'options' => [],
                ];
                continue;
            }

            if (preg_match('/^\s*(?:[-*]\s*)?([A-D])[.)]\s*(.+)$/i', $trim, $matches) && $current !== null) {
                $label = strtoupper($matches[1]);
                $current['options'][$label] = $this->remove_markdown_emphasis(trim($matches[2]));
                continue;
            }

            if ($current !== null && !empty($current['stem'])) {
                $current['stem'] .= ' ' . $this->remove_markdown_emphasis($trim);
            }
        }

        if ($current !== null) {
            $questions[] = $current;
        }

        return $questions;
    }

    /**
     * Parse answer key lines in format: "1. A".
     *
     * @param string $raw
     * @return array
     */
    private function parse_answer_key(string $raw): array {
        $result = [];
        $currentkey = null;
        foreach (explode("\n", $raw) as $line) {
            $trim = trim($line);
            if ($trim === '') {
                continue;
            }
            $clean = $this->remove_markdown_emphasis($trim);
            if (preg_match('/^\s*(\d+)\s*[.)-]\s*(.+)$/', $clean, $matches)) {
                $currentkey = (string)((int)$matches[1]);
                $result[$currentkey] = $this->normalize_answer_key_value((string)$matches[2]);
                continue;
            }

            if ($currentkey !== null) {
                $extra = preg_replace('/^\s*[-*]\s+/', '', $clean);
                $extra = trim((string)$extra);
                if ($extra !== '') {
                    $result[$currentkey] = $this->normalize_answer_key_value($result[$currentkey] . ' ' . $extra);
                }
            }
        }
        return $result;
    }

    /**
     * Return first non-empty line.
     *
     * @param string $raw
     * @return string
     */
    private function first_non_empty_line(string $raw): string {
        foreach (explode("\n", $raw) as $line) {
            if (trim($line) !== '') {
                return trim($line);
            }
        }
        return '';
    }

    /**
     * Collapse multi-line text into one readable line.
     *
     * @param string $raw
     * @return string
     */
    private function collapse_lines(string $raw): string {
        $parts = [];
        foreach (explode("\n", $raw) as $line) {
            $clean = trim($this->remove_markdown_emphasis($line));
            if ($clean !== '') {
                $parts[] = preg_replace('/^\s*(?:[-*]\s+|\d+[.)]\s+)/', '', $clean);
            }
        }
        return trim(implode(' ', $parts));
    }

    /**
     * Remove simple markdown emphasis symbols.
     *
     * @param string $text
     * @return string
     */
    private function remove_markdown_emphasis(string $text): string {
        $text = preg_replace('/[*_`]+/', '', $text);
        return trim((string)$text);
    }

    /**
     * Normalize answer-key value to keep only A/B/C/D when possible.
     *
     * @param string $raw
     * @return string
     */
    private function normalize_answer_key_value(string $raw): string {
        $value = str_replace("\xc2\xa0", ' ', (string)$raw);
        $value = trim($this->remove_markdown_emphasis($value));
        $upper = strtoupper($value);
        if (preg_match('/^(?:OPTION\\s+)?([ABCD])(?:[\\s\\).:\\-].*)?$/', $upper, $matches)) {
            return (string)$matches[1];
        }
        return $value;
    }
}
