#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

class HTMLToMarkdownConverter {
    constructor() { this.output = ''; }

    htmlToMarkdown(html) {
        // Strip scripts, styles, comments
        html = html.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gis, '');
        html = html.replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gis, '');
        html = html.replace(/<!--[\s\S]*?-->/g, '');
        // Strip nav elements
        html = html.replace(/<nav\b[\s\S]*?<\/nav>/gi, '');
        // Section wrappers
        html = html.replace(/<div[^>]*class=["']manuscript-section[^"']*["'][^>]*data-section=["']([^"']*)["'][^>]*>/gi, '\n\n## $1\n\n');
        // Headings
        html = html.replace(/<h1[^>]*>([\s\S]*?)<\/h1>/gi, '\n# $1\n\n');
        html = html.replace(/<h2[^>]*>([\s\S]*?)<\/h2>/gi, '\n## $1\n\n');
        html = html.replace(/<h3[^>]*>([\s\S]*?)<\/h3>/gi, '\n### $1\n\n');
        html = html.replace(/<h4[^>]*>([\s\S]*?)<\/h4>/gi, '\n#### $1\n\n');
        // Figures
        html = html.replace(/<figcaption[^>]*>([\s\S]*?)<\/figcaption>/gi, '\n\n    $1\n');
        html = html.replace(/<figure[^>]*>([\s\S]*?)<\/figure>/gi, '\n\n$1\n\n');
        
        // Images - convert to markdown syntax ![alt](src)
        html = html.replace(/<img[^>]+>/gi, (tag) => {
            const srcMatch = tag.match(/src=["']([^"']+)["']/i);
            const altMatch = tag.match(/alt=["']([^"']*)["']/i);
            const src = srcMatch ? srcMatch[1] : '';
            const alt = altMatch ? altMatch[1] : '';
            return src ? `\n![${alt}](${src})\n` : '';
        });

        // Block elements (use [\s\S] instead of . to handle multi-line)
        html = html.replace(/<blockquote[^>]*>([\s\S]*?)<\/blockquote>/gi, '\n> $1\n\n');
        html = html.replace(/<p[^>]*>([\s\S]*?)<\/p>/gi, '$1\n\n');
        // Inline formatting
        html = html.replace(/<(strong|b)[^>]*>([\s\S]*?)<\/(strong|b)>/gi, '**$2**');
        html = html.replace(/<(em|i)[^>]*>([\s\S]*?)<\/(em|i)>/gi, '*$2*');
        html = html.replace(/<code[^>]*>([\s\S]*?)<\/code>/gi, '`$1`');
        // Lists
        html = html.replace(/<li[^>]*>([\s\S]*?)<\/li>/gi, '- $1\n');
        // Line breaks
        html = html.replace(/<br\s*\/?>/gi, '\n');
        html = html.replace(/<hr\s*\/?>/gi, '\n---\n');
        // Strip all remaining tags
        html = html.replace(/<[^>]+>/g, '');
        // Decode common HTML entities
        html = html.replace(/&amp;/g, '&');
        html = html.replace(/&lt;/g, '<');
        html = html.replace(/&gt;/g, '>');
        html = html.replace(/&nbsp;/g, ' ');
        html = html.replace(/&mdash;/g, '—');
        html = html.replace(/&ndash;/g, '–');
        html = html.replace(/&#10004;/g, '✔');
        html = html.replace(/&#10008;/g, '✘');
        // Clean whitespace
        return html.replace(/\n{3,}/g, '\n\n').trim();
    }

    async convertSiteToMarkdown() {
        console.log('🔄 Converting HTML site to markdown (TEP-JWST)...');
        try {
            // Read manifest to get ordered component list
            const manifestPath = path.join(__dirname, 'manifest.json');
            if (!fs.existsSync(manifestPath)) throw new Error('manifest.json not found.');
            const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
            const sections = manifest.sections.sort((a, b) => a.order - b.order);

            // Read and concatenate all component HTML files directly
            let allHtml = '';
            for (const section of sections) {
                const componentPath = path.join(__dirname, 'components', section.file);
                if (fs.existsSync(componentPath)) {
                    const html = fs.readFileSync(componentPath, 'utf8');
                    allHtml += `\n<!-- SECTION: ${section.title} -->\n${html}\n`;
                    console.log(`  ✓ ${section.file} (${(html.length / 1024).toFixed(1)} KB)`);
                } else {
                    console.warn(`  ⚠ Missing: ${section.file}`);
                }
            }

            console.log(`  Total HTML: ${(allHtml.length / 1024).toFixed(1)} KB`);
            const markdown = "# Temporal Shear: Reconciling JWST's Impossible Galaxies\n\n" + this.htmlToMarkdown(allHtml);
            const outputPath = path.join(__dirname, '..', '13manuscript-tep-jwst.md');
            fs.writeFileSync(outputPath, markdown, 'utf8');
            console.log(`✅ Markdown saved to: ${outputPath} (${(markdown.length / 1024).toFixed(1)} KB)`);
        } catch (error) {
            console.error('❌ Markdown conversion failed:', error.message);
        }
    }
}

if (require.main === module) { const c = new HTMLToMarkdownConverter(); c.convertSiteToMarkdown(); }
module.exports = { HTMLToMarkdownConverter };
