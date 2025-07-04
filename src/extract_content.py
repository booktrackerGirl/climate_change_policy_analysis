## This code is to validate if the summary of the policies correctly represent the policy document

import requests, io, os, time, re, base64, gc, argparse, random, asyncio
import pandas as pd
import csv
import numpy as np
import fitz  # PyMuPDF
from io import BytesIO
from PIL import Image
import pytesseract
import pdfplumber
import psutil
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, urljoin
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import undetected_chromedriver as uc
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
import PyPDF2
import io
import dateparser
import torch
torch.cuda.empty_cache()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

import psutil

def clean_text(val):
    if isinstance(val, str):
        return val.encode("utf-8", errors="replace").decode("utf-8")
    return val

def kill_orphan_chrome():
    for proc in psutil.process_iter(["pid", "name"]):
        if proc.info["name"] and "chrome" in proc.info["name"].lower():
            try:
                proc.kill()
                print(f"Killed leftover Chrome process: {proc.info['pid']}")
            except:
                pass

def is_accessible(url, timeout=10):
    try:
        head = requests.head(url, allow_redirects=True, timeout=timeout)
        if head.status_code < 400:
            return True
        else:
            print(f"Inaccessible URL (HTTP {head.status_code}): {url}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"HEAD failed, trying GET: {e}")
        try:
            get = requests.get(url, stream=True, timeout=timeout)
            if get.status_code < 400:
                return True
            else:
                print(f"Inaccessible via GET (HTTP {get.status_code}): {url}")
                return False
        except Exception as e2:
            print(f"Both HEAD and GET failed for {url}: {e2}")
            return False


class PDFExtractor:
    def __init__(self):
        self.extraction_methods = [
            ("Direct PDF", self._extract_direct_pdf),
            ("Simple PDF Check", self._extract_simple_pdf),
            ("UNFFCC Extraction", self._extract_unfccc_pdf),
            ("Climate Laws Extraction", self._extract_climate_laws_pdf),
            ("Multi-purpose Extraction", self._extract_from_url_auto),
            ("Browser Extraction", self._extract_with_browser),
            ("HTML Page Content", self._extract_html_content),
            ("PDF Link Scan", self._extract_via_pdf_links),
            ("Playwright Live DOM", self._extract_pdf_live_dom),
            ("Advanced Browser Session", self._extract_with_advanced_browser),
            ("CAPTCHA-aware Extraction", self._extract_after_captcha)
        ]

    def _is_captcha_present(self, driver):
        try:
            page_text = driver.page_source.lower()
            return any(word in page_text for word in ["verify you are human", "captcha", "cloudflare"])
        except: return False

    def _extract_direct_pdf(self, url):
        try:
            # Try HEAD first to verify it's a PDF
            head = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=30)
            content_type = head.headers.get("Content-Type", "").lower()

            if "pdf" not in content_type:
                print(f"HEAD check did not confirm PDF. Trying GET instead for {url}")
        
            # GET full content
            response = requests.get(url, headers=HEADERS, stream=True, timeout=30)
            if response.status_code == 403:
                print("Access forbidden (403). Likely behind bot protection.")
                return None

            response.raise_for_status()

            if "pdf" not in response.headers.get("Content-Type", "").lower():
                print(f"Not a PDF: {response.headers.get('Content-Type')}")
                #return None
            
            with io.BytesIO(response.content) as f:
                with pdfplumber.open(f) as pdf:
                    text = "\n".join([p.extract_text() or "" for p in pdf.pages])
                    meta = pdf.metadata or {}
            return {"text": text.strip(), "metadata": {"pdf_url": url, "title": meta.get("Title"), "author": meta.get("Author"), "num_pages": len(pdf.pages)}}
        except: return None

    def _extract_simple_pdf(self, url):
        try:
            r = requests.get(url, timeout=15, stream =False)
            if r.status_code != 200:
                print("Not a successful response")
                return None

            content_type = r.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type:
                print("Content-Type is not PDF")
                #return None
            
            # check the PDF header
            initial_bytes = r.content[:2048]
            if b"%PDF" not in initial_bytes:
                print("PDF header not found in initial content")
                #return None
            
            # Reset the stream to re-read the full content
            r = requests.get(url, timeout=15)
            buffer = io.BytesIO(r.content)
            doc = fitz.open(stream=buffer, filetype="pdf")
            text = "\n".join([page.get_text() for page in doc])
            return {"text": text.strip(), "metadata": {"pdf_url": url, "title": None, "author": None, "num_pages": len(doc)}}
        except: return None
    
    def _extract_unfccc_pdf(self, url):
        """
        Extract PDF from UNFCCC website
    
        Args:
           url: UNFCCC PDF URL
    
        Returns:
            PDF content as bytes, or None if failed
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/pdf,*/*',
            'Referer': 'https://unfccc.int/'
        }
    
        session = requests.Session()
        session.headers.update(headers)
    
        # Try 3 times with different approaches
        for attempt in range(3):
            try:
                if attempt == 1:
                    # Visit base site first to establish session
                    session.get('https://unfccc.int', timeout=10)
            
                response = session.get(url, timeout=30)
                response.raise_for_status()
            
                # Check if it's actually a PDF
                content = response.content
                if content.startswith(b'%PDF') or 'pdf' in response.headers.get('content-type', ''):
                    
                    try: 
                        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                    
                        # Extract all text
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"

                        # Get metadata
                        meta = pdf_reader.metadata or {}

                        return {
                            "text": text.strip(),
                            "metadata": {
                            "pdf_url": response.url,
                            "title": meta.get("/Title"),
                            "author": meta.get("/Author"),
                            "num_pages": len(pdf_reader.pages)
                            }
                        }
            
                    except Exception as e:
                        print(f"Error processing PDF: {e}")
                        return None
    
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                    continue
                print(f"Failed to extract PDF: {e}")
        return None
    
    def _extract_climate_laws_pdf(self, url):
        """
        Simple PDF text extraction with OCR for scanned documents from Climate Laws domain.
        """
        try:
            # Get PDF content
            response = requests.get(url, timeout=30)
            response.raise_for_status()
       
            # Open PDF and extract text
            pdf_doc = fitz.open(stream=response.content, filetype="pdf")
            text_parts = []
       
            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
           
                # Try text extraction first
                text = page.get_text()
                if text.strip():
                    text_parts.append(text.strip())
                else:
                    # Use OCR for scanned pages
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                    img = Image.open(BytesIO(pix.tobytes("png")))
                    ocr_text = pytesseract.image_to_string(img, lang='eng')
                    if ocr_text.strip():
                        text_parts.append(ocr_text.strip())
       
            pdf_doc.close()
       
            return {
                "text": "\n".join(text_parts).strip(),
                "metadata": {
                   "pdf_url": response.url,
                   "title": None,
                    "author": None,
                    "num_pages": len(text_parts)
                  }
                }
       
        except Exception:
            return None

    def _extract_with_browser(self, url):
        try:
            options = Options()
            options.add_argument("--headless=new")
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            driver.get(url)
            time.sleep(3)
            final_url = driver.current_url
            r = requests.get(final_url, headers=HEADERS)
            with io.BytesIO(r.content) as f:
                with pdfplumber.open(f) as pdf:
                    text = "".join([p.extract_text() or "" for p in pdf.pages])
                    meta = pdf.metadata
            return {"text": text.strip(), "metadata": {"pdf_url": final_url, "title": meta.get("Title"), "author": meta.get("Author"), "num_pages": len(pdf.pages)}}
        except: return None
        finally:
            try: 
                driver.quit()
                del driver
            except: pass

    def _extract_html_content(self, url):
        async def extract_async():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=60000, wait_until="domcontentloaded")
                html = await page.content()
                await browser.close()
                return html
        try:
            try: 
                html = asyncio.run(extract_async())
            except RuntimeError:
                loop = asyncio.get_event_loop()
                html = loop.run_until_complete(extract_async())
            if not html:
                return None
            
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style"]): tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            title = soup.title.string.strip() if soup.title and soup.title.string else None

            return {"text": text, "metadata": {"pdf_url": url, "title": title, "author": None, "num_pages": None}}
        except: return None

    def _extract_via_pdf_links(self, url):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()

            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup.find_all(["a", "iframe", "embed", "object"]):
                href = tag.get("href") or tag.get("src") or tag.get("data")
                if href and "pdf" in href.lower():
                    full_url = urljoin(r.url, href)
                    return self._extract_direct_pdf(full_url)
        except Exception as e: 
            print(f"Error in _extract_via_pdf_links for {url}: {e}")
            return None

    def _extract_pdf_live_dom(self, url):

        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                page = browser.new_page()
                pdf_bytes = None
                captured_url = None

                # Intercept and capture any PDF response
                def handle_response(response):
                    nonlocal pdf_bytes, captured_url
                    try:
                        ct = response.headers.get("content-type", "")
                        if "pdf" in ct:
                            print(f"Captured PDF via network: {response.url}")
                            captured_url = response.url
                            pdf_bytes = response.body()
                    except:
                        pass

                page.on("response", handle_response)

                print(f"Visiting {url} via Playwright...")
                page.goto(url, timeout=60000)
                page.wait_for_timeout(5000)  # wait for JS to load embedded PDFs

                if pdf_bytes:
                    # Save and extract
                    with io.BytesIO(pdf_bytes) as f:
                        with pdfplumber.open(f) as pdf:
                            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                            metadata = pdf.metadata
                    return {
                    "text": text.strip(),
                    "metadata": {
                        "pdf_url": captured_url,
                        "title": metadata.get("Title"),
                        "author": metadata.get("Author"),
                        "num_pages": len(pdf.pages)
                    }
                }

                # Fallback: try to find PDF links in DOM
                print("Looking for PDF via DOM scanning...")
                for sel in ["iframe", "embed", "a"]:
                    elements = page.query_selector_all(sel)
                    for el in elements:
                        try:
                            src = el.get_attribute("src") or el.get_attribute("href")
                            if src and "pdf" in src.lower():
                                full_pdf_url = urljoin(url, src)
                                print(f"Found PDF in DOM: {full_pdf_url}")
                                return self._extract_direct_pdf(full_pdf_url)
                        except:
                            continue

                print("No PDF detected via DOM or network.")
                return None

        except Exception as e:
            print(f"Live DOM extraction failed: {e}")
            return None
        
    async def _extract_pdf_with_playwright_async(self, url):
        print(f"Visiting: {url}")
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=120000, wait_until="domcontentloaded")

                # Look for embedded PDF links
                for selector in ["iframe", "embed", "a"]:
                    elements = await page.query_selector_all(selector)
                    for el in elements:
                        href = await el.get_attribute("src") or await el.get_attribute("href")
                        if href and ".pdf" in href.lower():
                            full_url = urljoin(url, href)
                            print(f"Found PDF link: {full_url}")
                            await browser.close()
                            return self._extract_direct_pdf(full_url)

                # Fallback: extract raw text from the page
                html = await page.content()
                await browser.close()

                soup = BeautifulSoup(html, "lxml")
                for tag in soup(["script", "style"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)

                print("Extracted page text as fallback.")
                return {"text": text, "metadata": {"source_url": url}}

        except Exception as e:
            print(f"Failed in Playwright PDF async download: {e}")
            return None
        
    def _extract_from_url_auto(self, url):
        # Step 1: HEAD check
        try:
            head = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=10)
            content_type = head.headers.get("Content-Type", "").lower()
            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                return self._extract_direct_pdf(url)
        except Exception as e:
                print(f"HEAD request failed: {e}")

        # Step 2: Try full GET + check PDF
        try:
            r = requests.get(url, headers=HEADERS, timeout=10, stream=True)
            content_type = r.headers.get("Content-Type", "").lower()
            if "application/pdf" in content_type:
                return self._extract_direct_pdf(url)
        except:
            pass

        # Step 3: Playwright fallback
        return asyncio.run(self._extract_pdf_with_playwright_async(url))


    def _extract_with_advanced_browser(self, url):
        def create_driver():
            opts = uc.ChromeOptions()
            opts.add_argument("--headless=new")
            opts.add_argument("--disable-gpu")
            driver = uc.Chrome(options=opts)
            driver.set_page_load_timeout(60)
            return driver
        driver = create_driver()
        try:
            driver.get(url)
            time.sleep(3)
            if self._is_captcha_present(driver):
                start = time.time()
                while self._is_captcha_present(driver):
                    if time.time() - start > 150:
                        return None
                    time.sleep(20)
            session = requests.Session()
            for c in driver.get_cookies():
                session.cookies.set(c["name"], c["value"])
            headers = {"User-Agent": driver.execute_script("return navigator.userAgent;")}
            for _ in range(10):
                r = session.get(url, headers=headers)
                if "application/pdf" in r.headers.get("Content-Type", ""):
                    return self._extract_direct_pdf(url)
                time.sleep(5)
        except: return None
        finally:
            try: 
                driver.quit()
                del driver
            except: pass

    def _extract_after_captcha(self, url):
        try:
            options = uc.ChromeOptions()
            options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            driver = uc.Chrome(options=options)
            driver.set_page_load_timeout(60)
            driver = uc.Chrome(options=options)
            driver.get(url)
            time.sleep(5)

            # Wait up to 3 minutes for CAPTCHA to clear
            start = time.time()
            while "verify you are human" in driver.page_source.lower() or "just a moment" in driver.page_source.lower():
                elapsed = int(time.time() - start)
                if elapsed > 180:
                    print("CAPTCHA not cleared in 3 minutes.")
                    return None
                print(f"Waiting for CAPTCHA to clear... {elapsed}s")
                time.sleep(10)

            print("CAPTCHA cleared.")

            pdf_url = driver.current_url if driver.current_url.endswith(".pdf") else None
            if not pdf_url:
                for link in driver.find_elements(By.TAG_NAME, "a"):
                    href = link.get_attribute("href")
                    if href and ".pdf" in href:
                        pdf_url = href
                        break
            if not pdf_url:
                return None
            cookies = {c['name']: c['value'] for c in driver.get_cookies()}
            headers = {"User-Agent": driver.execute_script("return navigator.userAgent;")}
            r = requests.get(pdf_url, headers=headers, cookies=cookies)
            if "application/pdf" not in r.headers.get("Content-Type", "").lower():
                print("Downloaded content is not a PDF.")
                return None
            return self._extract_direct_pdf(pdf_url)
        
        except Exception as e: 
            print(f"CAPTCHA-aware extraction failed: {e}")
            return None
        finally:
            try: 
                driver.quit()
                del driver

            except: pass
    
    def extract_content(self, url):
        try:
            head = requests.head(url, timeout=10, allow_redirects=True)
            if head.status_code >= 400:
                print(f"Skipping (HTTP {head.status_code}): {url}")
                return np.nan
            if not is_accessible(url):
                print(f"Skipping inaccessible URL: {url}")
                return np.nan
        except Exception as e:
            print(f"HEAD request failed for {url}: {e}")
            return np.nan

        print(f"\nStarting extraction for: {url}")
        print("=" * 80)

        # 1. Direct PDF
        print(f"Trying Direct PDF for {url}")
        result = self._extract_direct_pdf(url)
        if result and result.get("text"):
            print("Extracted using Direct PDF")
            return result

        # 2. Simple PDF Check
        print(f"Trying Simple PDF Check for {url}")
        result = self._extract_simple_pdf(url)
        if result and result.get("text"):
            print("Extracted using Simple PDF Check")
            return result

        # 3. UNFCC Extraction
        print(f"Trying UNFCC Document Extraction for {url}")
        result = self._extract_unfccc_pdf(url)
        if result and result.get("text"):
            print("Extracted using UNFCC Extraction")
            return result
        
        # 3. Climate Laws Extraction
        print(f"Trying Climate Laws Document Extraction for {url}")
        result = self._extract_climate_laws_pdf(url)
        if result and result.get("text"):
            print("Extracted using Climate Laws PDF Extraction")
            return result
        
        
        # 4. Multi-purpose Check
        print(f"Trying Multi-purpose PDF Check for {url}")
        result = self._extract_from_url_auto(url)
        if result and result.get("text"):
            print("Extracted using Multi-purpose method Check")
            return result

        # 5. Browser Extraction
        print(f"Trying Browser Extraction for {url}")
        result = self._extract_with_browser(url)
        if result and result.get("text"):
            print("Extracted using Browser Extraction")
            return result

        # 6. HTML Page Content
        print(f"Trying HTML Page Content for {url}")
        result = self._extract_html_content(url)
        if result and result.get("text"):
            print("Extracted using HTML Page Content")
            return result

        # 7. PDF Link Scan
        print(f"Trying PDF Link Scan for {url}")
        result = self._extract_via_pdf_links(url)
        if result and result.get("text"):
            print("Extracted using PDF Link Scan")
            return result

        # 8. Playwright Live DOM
        print(f"Trying Playwright Live DOM for {url}")
        result = self._extract_pdf_live_dom(url)
        if result and result.get("text"):
            print("Extracted using Playwright Live DOM")
            return result

        # 9. Advanced Browser Session
        print(f"Trying Advanced Browser Session for {url}")
        result = self._extract_with_advanced_browser(url)
        if result and result.get("text"):
            print("Extracted using Advanced Browser Session")
            return result

        # 10. CAPTCHA-aware Extraction
        print(f"Trying CAPTCHA-aware Extraction for {url}")
        result = self._extract_after_captcha(url)
        if result and result.get("text"):
            print("Extracted using CAPTCHA-aware Extraction")
            return result

        print(f"All extraction methods failed for {url}")
        return np.nan



def check_memory_usage():
    if psutil.Process().memory_info().rss / 1024**2 > 800:
        gc.collect()
        print("Memory collected")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    df = pd.read_csv(f"{args.input_file}.csv")[4880:]
    df = df[~df["Document Content URL"].isna()]
    urls = df["Document Content URL"].tolist()

    extractor = PDFExtractor()
    texts = []
    for i, url in enumerate(urls, 1):
        print(f"\n{'='*60}\nProcessing {i}/{len(urls)}\n{url}\n{'='*60}")
        result = extractor.extract_content(url)
        if isinstance(result, dict) and "text" in result:
            texts.append(result["text"])
        else:
            texts.append(np.nan)
        
        # Clean up memory
        del result
        gc.collect()
        torch.cuda.empty_cache()
        check_memory_usage()
        time.sleep(3)
        if i % 10 == 0:
            data_temp = df[:i].copy()
            data_temp["Extracted Content"] = texts
            data_temp = data_temp.applymap(clean_text)
            data_temp.to_csv(
                f"{args.output_file}_checkpoint.csv",
                index=False,
                escapechar='\\',
                quoting=csv.QUOTE_NONNUMERIC,
                encoding='utf-8'
            )
            print(f"Saved checkpoint at {i}")
            kill_orphan_chrome()
            print("Cooling down for 60s...")
            time.sleep(60)

    df["Extracted Content"] = texts
    df.to_csv(f"{args.output_file}.csv", index=False,
                escapechar='\\',
                quoting=csv.QUOTE_NONNUMERIC,
                encoding='utf-8'
            )
    print("All done.")

if __name__ == "__main__":
    main()
